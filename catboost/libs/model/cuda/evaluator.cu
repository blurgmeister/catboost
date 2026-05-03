#include "evaluator.cuh"


#include <library/cpp/cuda/wrappers/kernel.cuh>
#include <library/cpp/cuda/wrappers/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.h>
#include <library/cpp/cuda/wrappers/kernel_helpers.cuh>

#include <util/string/cast.h>

#include <cuda_runtime.h>
#include <assert.h>

template<typename TFeatureType, TGPUDataInput::EFeatureLayout Layout>
struct TFeatureAccessor {
    TFeatureAccessor() = default;
    using TFeature = TFeatureType;
    using TFeaturePtr = const TFeature*;

    i32 Stride = 0;

    i32 FeatureCount = 0;
    i32 ObjectCount = 0;
    TFeaturePtr FeaturesPtr = nullptr;

    __forceinline__ __device__ TFeature operator()(i32 featureId, i32 objectId) const {
        if (Layout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId * Stride + objectId)
                : NegativeInfty();
        } else {
            return objectId < ObjectCount && featureId < FeatureCount ?
                __ldg(FeaturesPtr + featureId + objectId * Stride)
                : NegativeInfty();
        }
    }

    __forceinline__ __device__ int FeaturesCount() const {
        return FeatureCount;
    }

    __forceinline__ __device__ int SamplesCount() const {
        return ObjectCount;
    }
};

constexpr ui32 ObjectsPerThread = 4;
constexpr ui32 TreeSubBlockWidth = 8;
constexpr ui32 ExtTreeBlockWidth = 128;
constexpr ui32 QuantizationDocBlockSize = 256;

constexpr ui32 BlockWidth = 256;
constexpr ui32 EvalDocBlockSize = BlockWidth / TreeSubBlockWidth;

static_assert(EvalDocBlockSize >= WarpSize, "EvalBlockSize should be greater than WarpSize");
using TTreeIndex = uint4;

void TCudaQuantizedData::SetDimensions(ui32 effectiveBucketCount, ui32 objectsCount) {
    ObjectsCount = objectsCount;
    EffectiveBucketCount = effectiveBucketCount;
    const auto one32blockSize = WarpSize * effectiveBucketCount;
    const auto desiredQuantBuff = one32blockSize * NKernel::CeilDivide<ui32>(objectsCount, 128) * 4;
    if (BinarizedFeaturesBuffer.Size() < desiredQuantBuff) {
        BinarizedFeaturesBuffer = TCudaVec<TCudaQuantizationBucket>(desiredQuantBuff, NCuda::EMemoryType::Device);
    }
}

void TEvaluationDataCache::PrepareCopyBufs(size_t bufSize, size_t objectsCount) {
    if (CopyDataBufDevice.Size() < bufSize) {
        CopyDataBufDevice = TCudaVec<float>(AlignBy<2048>(bufSize), NCuda::EMemoryType::Device);
    }
    if (CopyDataBufHost.Size() < bufSize) {
        CopyDataBufHost = TCudaVec<float>(AlignBy<2048>(bufSize), NCuda::EMemoryType::Host);
    }
    if (ResultsFloatBuf.Size() < objectsCount) {
        ResultsFloatBuf = TCudaVec<float>(AlignBy<2048>(objectsCount), NCuda::EMemoryType::Device);
    }
    if (ResultsDoubleBuf.Size() < objectsCount) {
        ResultsDoubleBuf = TCudaVec<double>(AlignBy<2048>(objectsCount), NCuda::EMemoryType::Device);
    }
}

template<typename TFloatFeatureAccessor>
__launch_bounds__(QuantizationDocBlockSize, 1)
__global__ void Binarize(
    TFloatFeatureAccessor floatAccessor,
    const float* __restrict__ borders,
    const ui32* __restrict__ featureBorderOffsets,
    const ui32* __restrict__ featureBordersCount,
    const ui32* __restrict__ floatFeatureForBucketIdx,
    const ui32 bucketsCount,
    TCudaQuantizationBucket* __restrict__ target
) {
    const int blockby32 = blockIdx.x * QuantizationDocBlockSize / WarpSize + threadIdx.x / WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + threadIdx.x % WarpSize;

    const int targetBucketIdx = blockIdx.y;
    const float* featureBorders = borders + featureBorderOffsets[targetBucketIdx];
    const int featureBorderCount = __ldg(featureBordersCount + targetBucketIdx);
    const int featureIdx = floatFeatureForBucketIdx[targetBucketIdx];
    __shared__ float bordersLocal[QuantizationDocBlockSize];
    if (threadIdx.x < featureBorderCount) {
        bordersLocal[threadIdx.x] = __ldg(featureBorders + threadIdx.x);
    }
    __syncthreads();

    float4 features;
    features.x = floatAccessor(featureIdx, firstDocForThread + 0 * WarpSize);
    features.y = floatAccessor(featureIdx, firstDocForThread + 1 * WarpSize);
    features.z = floatAccessor(featureIdx, firstDocForThread + 2 * WarpSize);
    features.w = floatAccessor(featureIdx, firstDocForThread + 3 * WarpSize);

    TCudaQuantizationBucket bins = { 0 };
#pragma unroll 8
    for (int borderId = 0; borderId < featureBorderCount; ++borderId) {
        const float border = bordersLocal[borderId];

        bins.x += features.x > border;
        bins.y += features.y > border;
        bins.z += features.z > border;
        bins.w += features.w > border;
    }
    if (firstDocForThread < floatAccessor.SamplesCount()) {
        target[bucketsCount * WarpSize * blockby32 + targetBucketIdx * WarpSize + threadIdx.x % WarpSize] = bins;
    }
}

template<int TreeDepth>
TTreeIndex __device__ __forceinline__ CalcIndexesUnwrapped(const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    TTreeIndex result = { 0 };
#pragma unroll TreeDepth
    for (int depth = 0; depth < TreeDepth; ++depth) {
        const TGPURepackedBin bin = Ldg(curRepackedBinPtr + depth);
        TCudaQuantizationBucket buckets = __ldg(quantizedFeatures + bin.FeatureIdx);
        // |= operator fails (MLTOOLS-6839 on a100)
        result.x += ((buckets.x) >= bin.FeatureVal) << depth;
        result.y += ((buckets.y) >= bin.FeatureVal) << depth;
        result.z += ((buckets.z) >= bin.FeatureVal) << depth;
        result.w += ((buckets.w) >= bin.FeatureVal) << depth;
    }
    return result;
}

TTreeIndex __device__ CalcIndexesBase(int TreeDepth, const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    TTreeIndex bins = { 0 };
    for (int depth = 0; depth < TreeDepth; ++depth) {
        const TGPURepackedBin bin = Ldg(curRepackedBinPtr + depth);
        TCudaQuantizationBucket vals = __ldg(quantizedFeatures + bin.FeatureIdx);
        // |= operator fails (MLTOOLS-6839 on a100)
        bins.x += ((vals.x) >= bin.FeatureVal) << depth;
        bins.y += ((vals.y) >= bin.FeatureVal) << depth;
        bins.z += ((vals.z) >= bin.FeatureVal) << depth;
        bins.w += ((vals.w) >= bin.FeatureVal) << depth;
    }
    return bins;
}

TTreeIndex __device__ __forceinline__ CalcTreeVals(int curTreeDepth, const TGPURepackedBin* const __restrict__ curRepackedBinPtr, const TCudaQuantizationBucket* const __restrict__ quantizedFeatures) {
    switch (curTreeDepth) {
    case 6:
        return CalcIndexesUnwrapped<6>(curRepackedBinPtr, quantizedFeatures);
    case 7:
        return CalcIndexesUnwrapped<7>(curRepackedBinPtr, quantizedFeatures);
    case 8:
        return CalcIndexesUnwrapped<8>(curRepackedBinPtr, quantizedFeatures);
    default:
        return CalcIndexesBase(curTreeDepth, curRepackedBinPtr, quantizedFeatures);
    }
}

__device__ __forceinline__ ui8 GetQuantizedFeatureValueForDocument(
    const TCudaQuantizationBucket* const __restrict__ quantizedFeatures,
    ui32 bucketsCount,
    ui32 bucketIdx,
    ui32 docId
) {
    const ui32 docsPerBlock = WarpSize * ObjectsPerThread;
    const ui32 blockBy32 = docId / docsPerBlock;
    const ui32 laneId = docId % WarpSize;
    const ui32 docGroup = (docId / WarpSize) % ObjectsPerThread;
    const TCudaQuantizationBucket bins = __ldg(
        quantizedFeatures + bucketsCount * WarpSize * blockBy32 + bucketIdx * WarpSize + laneId
    );
    switch (docGroup) {
        case 0:
            return bins.x;
        case 1:
            return bins.y;
        case 2:
            return bins.z;
        default:
            return bins.w;
    }
}

__device__ __forceinline__ double CalcHardRightWeightForDocument(
    const TCudaQuantizationBucket* const __restrict__ quantizedFeatures,
    ui32 bucketsCount,
    const TGPURepackedBin& split,
    ui32 docId
) {
    const ui32 bucketIdx = split.FeatureIdx / WarpSize;
    return GetQuantizedFeatureValueForDocument(quantizedFeatures, bucketsCount, bucketIdx, docId) >= split.FeatureVal ? 1.0 : 0.0;
}

template <typename TFloatFeatureAccessor>
__device__ __forceinline__ double CalcInterpolatedRightWeightForDocument(
    TFloatFeatureAccessor floatAccessor,
    const TCudaQuantizationBucket* const __restrict__ quantizedFeatures,
    ui32 bucketsCount,
    const TGPURepackedBin& split,
    ui32 docId,
    const ui32* __restrict__ floatFeatureForBucketIdx,
    const ui32* __restrict__ bordersOffsets,
    const float* __restrict__ flatBorders,
    const double* __restrict__ interpolationSpans,
    ui32 interpolationSpansCount,
    const ui8* __restrict__ interpolationSpanModes,
    ui32 interpolationSpanModesCount,
    const double* __restrict__ interpolationMinSpans,
    ui32 interpolationMinSpansCount,
    bool useSigmoid,
    double minSpan
) {
    const ui32 bucketIdx = split.FeatureIdx / WarpSize;
    const ui32 floatFeatureIdx = __ldg(floatFeatureForBucketIdx + bucketIdx);
    if (floatFeatureIdx >= interpolationSpansCount) {
        return CalcHardRightWeightForDocument(quantizedFeatures, bucketsCount, split, docId);
    }

    const double configuredSpan = __ldg(interpolationSpans + floatFeatureIdx);
    if (configuredSpan <= 0.0) {
        return CalcHardRightWeightForDocument(quantizedFeatures, bucketsCount, split, docId);
    }

    const float rawValue = floatAccessor(floatFeatureIdx, docId);
    if (isnan(rawValue)) {
        return CalcHardRightWeightForDocument(quantizedFeatures, bucketsCount, split, docId);
    }

    const double border = __ldg(flatBorders + __ldg(bordersOffsets + bucketIdx) + split.FeatureVal - 1);
    const bool useRelativeSpan = floatFeatureIdx < interpolationSpanModesCount && __ldg(interpolationSpanModes + floatFeatureIdx) != 0;
    const double featureMinSpan = floatFeatureIdx < interpolationMinSpansCount
        ? __ldg(interpolationMinSpans + floatFeatureIdx)
        : minSpan;
    const double actualSpan = useRelativeSpan
        ? fmax(featureMinSpan, configuredSpan * fabs(border))
        : configuredSpan;
    if (actualSpan <= 0.0) {
        return CalcHardRightWeightForDocument(quantizedFeatures, bucketsCount, split, docId);
    }

    if (!useSigmoid) {
        const double clampedValue = fmin(fmax(static_cast<double>(rawValue), border - actualSpan), border + actualSpan);
        return (clampedValue - (border - actualSpan)) / (2.0 * actualSpan);
    }

    const double slope = 5.0 / actualSpan;
    return 1.0 / (1.0 + exp(-slope * (static_cast<double>(rawValue) - border)));
}

__launch_bounds__(BlockWidth, 1)
__global__ void EvalObliviousTrees(
    const TCudaQuantizationBucket* __restrict__ quantizedFeatures,
    const ui32* __restrict__ treeSizes,
    const ui32 treeStart,
    const ui32 treeEnd,
    const ui32* __restrict__ treeStartOffsets,
    const TGPURepackedBin* __restrict__ repackedBins,
    const ui32* __restrict__ firstLeafOfset,
    const ui32 bucketsCount,
    const TCudaEvaluatorLeafType* __restrict__ leafValues,
    const ui32 documentCount,
    TCudaEvaluatorLeafType* __restrict__ results) {

    const int innerBlockBy32 = threadIdx.x / WarpSize;
    const int blockby32 = blockIdx.y * EvalDocBlockSize / WarpSize + innerBlockBy32;
    const int inBlockId = threadIdx.x % WarpSize;
    const int firstDocForThread = blockby32 * WarpSize * ObjectsPerThread + inBlockId;

    quantizedFeatures += bucketsCount * WarpSize * blockby32 + threadIdx.x % WarpSize;

    const int firstTreeIdx = treeStart + TreeSubBlockWidth * ExtTreeBlockWidth * (threadIdx.y + TreeSubBlockWidth * blockIdx.x);
    const int lastTreeIdx = min(firstTreeIdx + TreeSubBlockWidth * ExtTreeBlockWidth, static_cast<int>(treeEnd));
    double4 localResult = { 0 };

    if (firstTreeIdx < lastTreeIdx && firstDocForThread < documentCount) {
        const TGPURepackedBin* __restrict__ curRepackedBinPtr = repackedBins + __ldg(treeStartOffsets + firstTreeIdx);
        leafValues += firstLeafOfset[firstTreeIdx];
        int treeIdx = firstTreeIdx;
        const int lastTreeBy2 = lastTreeIdx - ((lastTreeIdx - firstTreeIdx) & 0x3);
        for (; treeIdx < lastTreeBy2; treeIdx += 2) {
            const int curTreeDepth1 = __ldg(treeSizes + treeIdx);
            const int curTreeDepth2 = __ldg(treeSizes + treeIdx + 1);

            const TTreeIndex bins1 = CalcTreeVals(curTreeDepth1, curRepackedBinPtr, quantizedFeatures);
            const TTreeIndex bins2 = CalcTreeVals(curTreeDepth2, curRepackedBinPtr + curTreeDepth1, quantizedFeatures);
            const auto leafValues2 = leafValues + (1 << curTreeDepth1);
            localResult.x += __ldg(leafValues + bins1.x) + __ldg(leafValues2 + bins2.x);
            localResult.y += __ldg(leafValues + bins1.y) + __ldg(leafValues2 + bins2.y);
            localResult.z += __ldg(leafValues + bins1.z) + __ldg(leafValues2 + bins2.z);
            localResult.w += __ldg(leafValues + bins1.w) + __ldg(leafValues2 + bins2.w);
            curRepackedBinPtr += curTreeDepth1 + curTreeDepth2;
            leafValues = leafValues2 + (1 << curTreeDepth2);
        }
        for (; treeIdx < lastTreeIdx; ++treeIdx) {
            const int curTreeDepth = __ldg(treeSizes + treeIdx);
            const TTreeIndex bins = CalcTreeVals(curTreeDepth, curRepackedBinPtr, quantizedFeatures);
            localResult.x += __ldg(leafValues + bins.x);
            localResult.y += __ldg(leafValues + bins.y);
            localResult.z += __ldg(leafValues + bins.z);
            localResult.w += __ldg(leafValues + bins.w);
            curRepackedBinPtr += curTreeDepth;
            leafValues += (1 << curTreeDepth);
        }
    }
    // TODO(kirillovs): reduce code is valid if those conditions met
    static_assert(EvalDocBlockSize * ObjectsPerThread == 128, "");
    static_assert(EvalDocBlockSize == 32, "");
    __shared__ TCudaEvaluatorLeafType reduceVals[EvalDocBlockSize * ObjectsPerThread * TreeSubBlockWidth];
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 0 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.x;
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 1 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.y;
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 2 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.z;
    reduceVals[innerBlockBy32 * WarpSize * ObjectsPerThread + WarpSize * 3 + inBlockId + threadIdx.y * EvalDocBlockSize * ObjectsPerThread] = localResult.w;
    __syncthreads();
    TCudaEvaluatorLeafType lr = reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize];
    for (int i = 256; i < 256 * 4; i += 256) {
        lr += reduceVals[i + threadIdx.x + threadIdx.y * EvalDocBlockSize];
    }
    reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize] = lr;
    __syncthreads();
    if (threadIdx.y < ObjectsPerThread) {
        TAtomicAdd<TCudaEvaluatorLeafType>::Add(
            results + blockby32 * WarpSize * ObjectsPerThread + threadIdx.x + threadIdx.y * EvalDocBlockSize,
            reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize] + reduceVals[threadIdx.x + threadIdx.y * EvalDocBlockSize + 128]
        );
    }
}

template <typename TFloatFeatureAccessor>
__global__ void EvalObliviousTreesWithInterpolation(
    TFloatFeatureAccessor floatAccessor,
    const TCudaQuantizationBucket* __restrict__ quantizedFeatures,
    const ui32* __restrict__ treeSizes,
    const ui32* __restrict__ treeStartOffsets,
    const TGPURepackedBin* __restrict__ repackedBins,
    const ui32* __restrict__ firstLeafOffset,
    const ui32 bucketsCount,
    const TCudaEvaluatorLeafType* __restrict__ leafValues,
    const ui32* __restrict__ floatFeatureForBucketIdx,
    const ui32* __restrict__ bordersOffsets,
    const float* __restrict__ flatBorders,
    const double* __restrict__ interpolationSpans,
    const ui32 interpolationSpansCount,
    const ui8* __restrict__ interpolationSpanModes,
    const ui32 interpolationSpanModesCount,
    const double* __restrict__ interpolationMinSpans,
    const ui32 interpolationMinSpansCount,
    const bool useSigmoid,
    const double minSpan,
    const ui32 treeStart,
    const ui32 treeEnd,
    const ui32 documentCount,
    const ui32 approxDimension,
    TCudaEvaluatorLeafType* __restrict__ results
) {
    const ui32 docId = blockIdx.x * blockDim.x + threadIdx.x;
    if (docId >= documentCount) {
        return;
    }

    TCudaEvaluatorLeafType* docResult = results + docId * approxDimension;
    for (ui32 treeId = treeStart; treeId < treeEnd; ++treeId) {
        const int depth = __ldg(treeSizes + treeId);
        const ui32 splitOffset = __ldg(treeStartOffsets + treeId);
        const ui32 leafOffset = __ldg(firstLeafOffset + treeId);
        const ui32 leafCount = 1u << depth;
        double rightWeights[64];

        for (int currentDepth = 0; currentDepth < depth; ++currentDepth) {
            rightWeights[currentDepth] = CalcInterpolatedRightWeightForDocument(
                floatAccessor,
                quantizedFeatures,
                bucketsCount,
                Ldg(repackedBins + splitOffset + currentDepth),
                docId,
                floatFeatureForBucketIdx,
                bordersOffsets,
                flatBorders,
                interpolationSpans,
                interpolationSpansCount,
                interpolationSpanModes,
                interpolationSpanModesCount,
                interpolationMinSpans,
                interpolationMinSpansCount,
                useSigmoid,
                minSpan
            );
        }

        double leafWeights[1024];
        leafWeights[0] = 1.0;
        for (int currentDepth = 0; currentDepth < depth; ++currentDepth) {
            const double rw = rightWeights[currentDepth];
            const double lw = 1.0 - rw;
            const ui32 currentLeafCount = 1u << currentDepth;
            for (ui32 leafIdx = 0; leafIdx < currentLeafCount; ++leafIdx) {
                leafWeights[leafIdx + currentLeafCount] = leafWeights[leafIdx] * rw;
                leafWeights[leafIdx] *= lw;
            }
        }

        for (ui32 leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            const double leafWeight = leafWeights[leafIdx];
            if (leafWeight == 0.0) {
                continue;
            }
            const TCudaEvaluatorLeafType* leafValuePtr = leafValues + (leafOffset + leafIdx) * approxDimension;
            for (ui32 dim = 0; dim < approxDimension; ++dim) {
                docResult[dim] += __ldg(leafValuePtr + dim) * leafWeight;
            }
        }
    }
}

template<NCB::NModelEvaluation::EPredictionType PredictionType, bool OneDimension>
__global__ void ProcessResultsImpl(
    const float* __restrict__ rawResults,
    ui32 resultsSize,
    const double* __restrict__ bias,
    double scale,
    double* hostMemResults,
    ui32 approxDimension
) {
    for (ui32 resultId = threadIdx.x; resultId < resultsSize; resultId += blockDim.x) {
        if (OneDimension) {
            double res = scale * __ldg(rawResults + resultId) + __ldg(bias);
            if (PredictionType == NCB::NModelEvaluation::EPredictionType::RawFormulaVal) {
                hostMemResults[resultId] = res;
            } else if (PredictionType == NCB::NModelEvaluation::EPredictionType::Probability) {
                hostMemResults[resultId] = 1 / (1 + exp(-res));
            } else if (PredictionType == NCB::NModelEvaluation::EPredictionType::Class) {
                hostMemResults[resultId] = res > 0;
            } else {
                assert(0);
            }
        } else {
            const float* rawResultsSub = rawResults + resultId * approxDimension;
            if (PredictionType == NCB::NModelEvaluation::EPredictionType::Class) {
                double maxVal = scale * __ldg(rawResultsSub) + __ldg(bias);
                ui32 maxPos = 0;
                for (ui32 dim = 1; dim < approxDimension; ++dim) {
                    double val = scale * __ldg(rawResultsSub + dim) + __ldg(bias + dim);
                    if (val > maxVal) {
                        maxVal = val;
                        maxPos = dim;
                    }
                }
                hostMemResults[resultId] = maxPos;
            } else {
                double* hostMemResultsBase = hostMemResults + resultId * approxDimension;

                for (ui32 dim = 0; dim < approxDimension; ++dim) {
                    hostMemResultsBase[dim] = scale * __ldg(rawResultsSub + dim) + __ldg(bias + dim);
                }

                if (PredictionType != NCB::NModelEvaluation::EPredictionType::RawFormulaVal) {
                    // TODO(kirillovs): write softmax
                    assert(0);
                }
            }
        }
    }
}

template<bool OneDimension>
void ProcessResults(
    const TGPUCatboostEvaluationContext& ctx,
    NCB::NModelEvaluation::EPredictionType predictionType,
    size_t objectsCount) {

    switch (predictionType) {
        case NCB::NModelEvaluation::EPredictionType::RawFormulaVal:
            ProcessResultsImpl<NCB::NModelEvaluation::EPredictionType::RawFormulaVal, OneDimension><<<1, 256, 0, ctx.Stream>>> (
                ctx.EvalDataCache.ResultsFloatBuf.Get(),
                objectsCount,
                ctx.GPUModelData.Bias.Get(),
                ctx.GPUModelData.Scale,
                ctx.EvalDataCache.ResultsDoubleBuf.Get(),
                ctx.GPUModelData.ApproxDimension
            );
            break;
        case NCB::NModelEvaluation::EPredictionType::Exponent:
        case NCB::NModelEvaluation::EPredictionType::RMSEWithUncertainty:
        case NCB::NModelEvaluation::EPredictionType::MultiProbability:
            ythrow yexception() << "Unimplemented on GPU: prediction type " << ToString(predictionType);
            break;
        case NCB::NModelEvaluation::EPredictionType::Probability:
            ProcessResultsImpl<NCB::NModelEvaluation::EPredictionType::Probability, OneDimension><<<1, 256, 0, ctx.Stream>>> (
                ctx.EvalDataCache.ResultsFloatBuf.Get(),
                objectsCount,
                ctx.GPUModelData.Bias.Get(),
                ctx.GPUModelData.Scale,
                ctx.EvalDataCache.ResultsDoubleBuf.Get(),
                ctx.GPUModelData.ApproxDimension
            );
            break;
        case NCB::NModelEvaluation::EPredictionType::Class:
            ProcessResultsImpl<NCB::NModelEvaluation::EPredictionType::Class, OneDimension><<<1, 256, 0, ctx.Stream>>> (
                ctx.EvalDataCache.ResultsFloatBuf.Get(),
                objectsCount,
                ctx.GPUModelData.Bias.Get(),
                ctx.GPUModelData.Scale,
                ctx.EvalDataCache.ResultsDoubleBuf.Get(),
                ctx.GPUModelData.ApproxDimension
            );
            break;
    }
}


void TGPUCatboostEvaluationContext::EvalQuantizedData(
    const TCudaQuantizedData* data,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> result,
    NCB::NModelEvaluation::EPredictionType predictionType
    ) const {
    if (GPUModelData.InterpolationEnabled && data->RawFloatFeatureCount != 0) {
        ClearMemoryAsync(EvalDataCache.ResultsFloatBuf.AsArrayRef(), Stream);
        const ui32 blockSize = 256;
        const ui32 gridSize = NKernel::CeilDivide<ui32>(data->GetObjectsCount(), blockSize);
        if (data->RawFloatRowFirst) {
            TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::RowFirst> floatFeatureAccessor;
            floatFeatureAccessor.FeatureCount = data->RawFloatFeatureCount;
            floatFeatureAccessor.Stride = data->RawFloatStride;
            floatFeatureAccessor.ObjectCount = data->GetObjectsCount();
            floatFeatureAccessor.FeaturesPtr = data->RawFloatData.Get();
            EvalObliviousTreesWithInterpolation<<<gridSize, blockSize, 0, Stream>>>(
                floatFeatureAccessor,
                data->BinarizedFeaturesBuffer.Get(),
                GPUModelData.TreeSizes.Get(),
                GPUModelData.TreeStartOffsets.Get(),
                GPUModelData.TreeSplits.Get(),
                GPUModelData.TreeFirstLeafOffsets.Get(),
                GPUModelData.FloatFeatureForBucketIdx.Size(),
                GPUModelData.ModelLeafs.Get(),
                GPUModelData.FloatFeatureForBucketIdx.Get(),
                GPUModelData.BordersOffsets.Get(),
                GPUModelData.FlatBordersVector.Get(),
                GPUModelData.FloatFeatureInterpolationSpans.Get(),
                GPUModelData.FloatFeatureInterpolationSpans.Size(),
                GPUModelData.FloatFeatureInterpolationSpanModes.Get(),
                GPUModelData.FloatFeatureInterpolationSpanModes.Size(),
                GPUModelData.FloatFeatureInterpolationMinSpans.Get(),
                GPUModelData.FloatFeatureInterpolationMinSpans.Size(),
                GPUModelData.InterpolationUseSigmoid,
                GPUModelData.InterpolationMinSpan,
                treeStart,
                treeEnd,
                data->GetObjectsCount(),
                GPUModelData.ApproxDimension,
                EvalDataCache.ResultsFloatBuf.Get()
            );
        } else {
            TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::ColumnFirst> floatFeatureAccessor;
            floatFeatureAccessor.FeatureCount = data->RawFloatFeatureCount;
            floatFeatureAccessor.Stride = data->RawFloatStride;
            floatFeatureAccessor.ObjectCount = data->GetObjectsCount();
            floatFeatureAccessor.FeaturesPtr = data->RawFloatData.Get();
            EvalObliviousTreesWithInterpolation<<<gridSize, blockSize, 0, Stream>>>(
                floatFeatureAccessor,
                data->BinarizedFeaturesBuffer.Get(),
                GPUModelData.TreeSizes.Get(),
                GPUModelData.TreeStartOffsets.Get(),
                GPUModelData.TreeSplits.Get(),
                GPUModelData.TreeFirstLeafOffsets.Get(),
                GPUModelData.FloatFeatureForBucketIdx.Size(),
                GPUModelData.ModelLeafs.Get(),
                GPUModelData.FloatFeatureForBucketIdx.Get(),
                GPUModelData.BordersOffsets.Get(),
                GPUModelData.FlatBordersVector.Get(),
                GPUModelData.FloatFeatureInterpolationSpans.Get(),
                GPUModelData.FloatFeatureInterpolationSpans.Size(),
                GPUModelData.FloatFeatureInterpolationSpanModes.Get(),
                GPUModelData.FloatFeatureInterpolationSpanModes.Size(),
                GPUModelData.FloatFeatureInterpolationMinSpans.Get(),
                GPUModelData.FloatFeatureInterpolationMinSpans.Size(),
                GPUModelData.InterpolationUseSigmoid,
                GPUModelData.InterpolationMinSpan,
                treeStart,
                treeEnd,
                data->GetObjectsCount(),
                GPUModelData.ApproxDimension,
                EvalDataCache.ResultsFloatBuf.Get()
            );
        }

        if (GPUModelData.ApproxDimension == 1) {
            ProcessResults<true>(*this, predictionType, data->GetObjectsCount());
        } else {
            ProcessResults<false>(*this, predictionType, data->GetObjectsCount());
        }

        NCuda::MemoryCopyAsync<double>(
            EvalDataCache.ResultsDoubleBuf.Slice(0, data->GetObjectsCount() * GPUModelData.ApproxDimension),
            result,
            Stream
        );
        return;
    }

    const dim3 treeCalcDimBlock(EvalDocBlockSize, TreeSubBlockWidth);
    const dim3 treeCalcDimGrid(
        NKernel::CeilDivide<unsigned int>(treeEnd - treeStart, TreeSubBlockWidth * ExtTreeBlockWidth),
        NKernel::CeilDivide<unsigned int>(data->GetObjectsCount(), EvalDocBlockSize * ObjectsPerThread)
    );
    ClearMemoryAsync(EvalDataCache.ResultsFloatBuf.AsArrayRef(), Stream);
    EvalObliviousTrees<<<treeCalcDimGrid, treeCalcDimBlock, 0, Stream>>> (
        data->BinarizedFeaturesBuffer.Get(),
        GPUModelData.TreeSizes.Get(),
        treeStart,
        treeEnd,
        GPUModelData.TreeStartOffsets.Get(),
        GPUModelData.TreeSplits.Get(),
        GPUModelData.TreeFirstLeafOffsets.Get(),
        GPUModelData.FloatFeatureForBucketIdx.Size(),
        GPUModelData.ModelLeafs.Get(),
        data->GetObjectsCount(),
        EvalDataCache.ResultsFloatBuf.Get()
    );

    if (GPUModelData.ApproxDimension == 1) {
        ProcessResults<true>(*this, predictionType, data->GetObjectsCount());
    } else {
        ProcessResults<false>(*this, predictionType, data->GetObjectsCount());
    }

    NCuda::MemoryCopyAsync<double>(EvalDataCache.ResultsDoubleBuf.Slice(0, data->GetObjectsCount()), result, Stream);
}

void TGPUCatboostEvaluationContext::QuantizeData(const TGPUDataInput& dataInput, TCudaQuantizedData* quantizedData) const{
    const dim3 quantizationDimBlock(QuantizationDocBlockSize, 1);
    const dim3 quantizationDimGrid(
        NKernel::CeilDivide<unsigned int>(dataInput.ObjectCount, QuantizationDocBlockSize * ObjectsPerThread),
        GPUModelData.BordersCount.Size() // float features from models
    );
    if (dataInput.FloatFeatureLayout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
        TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::ColumnFirst> floatFeatureAccessor;
        floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
        floatFeatureAccessor.Stride = dataInput.Stride;
        floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
        floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.data();
        Binarize<<<quantizationDimGrid, quantizationDimBlock, 0, Stream>>> (
            floatFeatureAccessor,
            GPUModelData.FlatBordersVector.Get(),
            GPUModelData.BordersOffsets.Get(),
            GPUModelData.BordersCount.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Size(),
            quantizedData->BinarizedFeaturesBuffer.Get()
        );
    } else {
        TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::RowFirst> floatFeatureAccessor;
        floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
        floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
        floatFeatureAccessor.Stride = dataInput.Stride;
        floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.data();
        Binarize<<<quantizationDimGrid, quantizationDimBlock, 0, Stream>>> (
            floatFeatureAccessor,
            GPUModelData.FlatBordersVector.Get(),
            GPUModelData.BordersOffsets.Get(),
            GPUModelData.BordersCount.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Size(),
            quantizedData->BinarizedFeaturesBuffer.Get()
        );
    }
}

void TGPUCatboostEvaluationContext::EvalData(
    const TGPUDataInput& dataInput,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> result,
    NCB::NModelEvaluation::EPredictionType predictionType) const {
    TCudaQuantizedData quantizedData;
    quantizedData.SetDimensions(GPUModelData.FloatFeatureForBucketIdx.Size(), dataInput.ObjectCount);
    QuantizeData(dataInput, &quantizedData);
    if (!GPUModelData.InterpolationEnabled) {
        EvalQuantizedData(&quantizedData, treeStart, treeEnd, result, predictionType);
        return;
    }

    ClearMemoryAsync(EvalDataCache.ResultsFloatBuf.AsArrayRef(), Stream);
    const ui32 blockSize = 256;
    const ui32 gridSize = NKernel::CeilDivide<ui32>(dataInput.ObjectCount, blockSize);
    if (dataInput.FloatFeatureLayout == TGPUDataInput::EFeatureLayout::ColumnFirst) {
        TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::ColumnFirst> floatFeatureAccessor;
        floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
        floatFeatureAccessor.Stride = dataInput.Stride;
        floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
        floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.data();
        EvalObliviousTreesWithInterpolation<<<gridSize, blockSize, 0, Stream>>>(
            floatFeatureAccessor,
            quantizedData.BinarizedFeaturesBuffer.Get(),
            GPUModelData.TreeSizes.Get(),
            GPUModelData.TreeStartOffsets.Get(),
            GPUModelData.TreeSplits.Get(),
            GPUModelData.TreeFirstLeafOffsets.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Size(),
            GPUModelData.ModelLeafs.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Get(),
            GPUModelData.BordersOffsets.Get(),
            GPUModelData.FlatBordersVector.Get(),
            GPUModelData.FloatFeatureInterpolationSpans.Get(),
            GPUModelData.FloatFeatureInterpolationSpans.Size(),
            GPUModelData.FloatFeatureInterpolationSpanModes.Get(),
            GPUModelData.FloatFeatureInterpolationSpanModes.Size(),
            GPUModelData.FloatFeatureInterpolationMinSpans.Get(),
            GPUModelData.FloatFeatureInterpolationMinSpans.Size(),
            GPUModelData.InterpolationUseSigmoid,
            GPUModelData.InterpolationMinSpan,
            treeStart,
            treeEnd,
            dataInput.ObjectCount,
            GPUModelData.ApproxDimension,
            EvalDataCache.ResultsFloatBuf.Get()
        );
    } else {
        TFeatureAccessor<float, TGPUDataInput::EFeatureLayout::RowFirst> floatFeatureAccessor;
        floatFeatureAccessor.FeatureCount = dataInput.FloatFeatureCount;
        floatFeatureAccessor.Stride = dataInput.Stride;
        floatFeatureAccessor.ObjectCount = dataInput.ObjectCount;
        floatFeatureAccessor.FeaturesPtr = dataInput.FlatFloatsVector.data();
        EvalObliviousTreesWithInterpolation<<<gridSize, blockSize, 0, Stream>>>(
            floatFeatureAccessor,
            quantizedData.BinarizedFeaturesBuffer.Get(),
            GPUModelData.TreeSizes.Get(),
            GPUModelData.TreeStartOffsets.Get(),
            GPUModelData.TreeSplits.Get(),
            GPUModelData.TreeFirstLeafOffsets.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Size(),
            GPUModelData.ModelLeafs.Get(),
            GPUModelData.FloatFeatureForBucketIdx.Get(),
            GPUModelData.BordersOffsets.Get(),
            GPUModelData.FlatBordersVector.Get(),
            GPUModelData.FloatFeatureInterpolationSpans.Get(),
            GPUModelData.FloatFeatureInterpolationSpans.Size(),
            GPUModelData.FloatFeatureInterpolationSpanModes.Get(),
            GPUModelData.FloatFeatureInterpolationSpanModes.Size(),
            GPUModelData.FloatFeatureInterpolationMinSpans.Get(),
            GPUModelData.FloatFeatureInterpolationMinSpans.Size(),
            GPUModelData.InterpolationUseSigmoid,
            GPUModelData.InterpolationMinSpan,
            treeStart,
            treeEnd,
            dataInput.ObjectCount,
            GPUModelData.ApproxDimension,
            EvalDataCache.ResultsFloatBuf.Get()
        );
    }

    if (GPUModelData.ApproxDimension == 1) {
        ProcessResults<true>(*this, predictionType, dataInput.ObjectCount);
    } else {
        ProcessResults<false>(*this, predictionType, dataInput.ObjectCount);
    }

    NCuda::MemoryCopyAsync<double>(EvalDataCache.ResultsDoubleBuf.Slice(0, dataInput.ObjectCount * GPUModelData.ApproxDimension), result, Stream);
}
