#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/model_quantization_adapter.h>
#include <catboost/private/libs/text_features/ut/lib/text_features_data.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ymath.h>

using namespace NCB;
using namespace NCB::NModelEvaluation;

const TVector<TVector<float>> DATA = {
    {0.f, 0.f, 0.f},
    {3.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {3.f, 1.f, 0.f},
    {0.f, 0.f, 1.f},
    {3.f, 0.f, 1.f},
    {0.f, 1.f, 1.f},
    {3.f, 1.f, 1.f},
};

TVector<TConstArrayRef<float>> GetFeatureRef(const TVector<TVector<float>>& data) {
    TVector<TConstArrayRef<float>> features(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        features[i] = data[i];
    };
    return features;
}

const auto FLOAT_FEATURES = GetFeatureRef(DATA);

TDataProviderPtr CreateBinaryClassificationInterpolationLeakPool() {
    static const TVector<float> WORST_RADIUS = {
        25.38f, 24.99f, 23.57f, 14.91f, 22.54f, 15.47f, 22.88f, 17.06f,
        15.49f, 15.09f, 19.19f, 20.42f, 20.96f, 16.84f, 15.03f, 17.46f,
        19.07f, 20.96f, 27.32f, 15.11f, 14.5f, 10.23f, 18.07f, 29.17f,
        26.46f, 22.25f, 17.62f, 21.31f, 20.27f, 20.01f, 23.15f, 16.82f,
        20.88f, 24.15f, 20.21f, 20.01f, 15.89f, 13.3f, 14.99f, 15.53f,
        15.93f, 12.84f, 24.09f, 17.38f, 16.23f, 22.82f, 8.964f, 15.67f,
        13.76f, 15.15f, 12.98f, 14.67f, 13.1f, 20.6f, 18.1f, 12.84f
    };
    static const TVector<float> MEAN_CONCAVITY = {
        0.3001f, 0.0869f, 0.1974f, 0.2414f, 0.198f, 0.1578f, 0.1127f, 0.09366f,
        0.1859f, 0.2273f, 0.03299f, 0.09954f, 0.2065f, 0.09938f, 0.2128f, 0.1639f,
        0.07395f, 0.1722f, 0.1479f, 0.06664f, 0.04568f, 0.02956f, 0.2077f, 0.1097f,
        0.1525f, 0.2229f, 0.1425f, 0.149f, 0.1683f, 0.09875f, 0.2319f, 0.1218f,
        0.2417f, 0.1657f, 0.1354f, 0.1348f, 0.1319f, 0.02562f, 0.02398f, 0.1063f,
        0.0311f, 0.1044f, 0.2107f, 0.09847f, 0.08259f, 0.1974f, 0.01588f, 0.1226f,
        0.06592f, 0.04751f, 0.01657f, 0.01857f, 0.01972f, 0.1772f, 0.05253f, 0.03036f
    };
    static const TVector<float> TARGET = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 1.f
    };

    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Boolean;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                2,
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{}
            );

            visitor->Start(metaInfo, WORST_RADIUS.size(), EObjectsOrder::Ordered, {});
            visitor->AddFloatFeature(
                0,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(WORST_RADIUS))
            );
            visitor->AddFloatFeature(
                1,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(MEAN_CONCAVITY))
            );
            visitor->AddTarget(
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(TARGET))
            );
            visitor->Finish();
        }
    );
}

TDataProviderPtr CreateRegressionInterpolationLeakPool() {
    static const TVector<float> MED_INC = {
        8.3252f, 8.3014f, 7.2574f, 5.6431f, 3.8462f, 4.0368f, 3.6591f, 3.12f,
        2.0804f, 3.6912f, 3.2031f, 3.2705f, 3.075f, 2.6736f, 1.9167f, 2.125f,
        2.775f, 2.1202f, 1.9911f, 2.6033f, 1.3578f, 1.7135f, 1.725f, 2.1806f,
        2.6f, 2.4038f, 2.4597f, 1.808f, 1.6424f, 1.6875f, 1.9274f, 1.9615f
    };
    static const TVector<float> AVE_ROOMS = {
        6.984126984126984f, 6.238137082601054f, 8.288135593220339f, 5.817351598173516f,
        6.281853281853282f, 4.761658031088083f, 4.9319066147859925f, 4.797527047913447f,
        4.294117647058823f, 4.970588235294118f, 5.477611940298507f, 4.772479564032698f,
        5.322649572649572f, 4.0f, 4.262903225806451f, 4.242424242424242f,
        5.9395770392749245f, 4.052805280528053f, 5.343675417661098f, 5.465454545454546f,
        4.524096385542169f, 4.478142076502732f, 5.096234309623431f, 5.193846153846154f,
        5.270142180094787f, 4.495798319327731f, 4.728033472803348f, 4.780856423173804f,
        4.40169133192389f, 4.703225806451613f, 5.068783068783069f, 4.882086167800454f
    };
    static const TVector<float> TARGET = {
        4.526f, 3.585f, 3.521f, 3.413f, 3.422f, 2.697f, 2.992f, 2.414f,
        2.267f, 2.611f, 2.815f, 2.418f, 2.135f, 1.913f, 1.592f, 1.4f,
        1.525f, 1.555f, 1.587f, 1.629f, 1.475f, 1.598f, 1.139f, 0.997f,
        1.326f, 1.075f, 0.938f, 1.055f, 1.089f, 1.32f, 1.223f, 1.152f
    };

    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                2,
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{}
            );

            visitor->Start(metaInfo, MED_INC.size(), EObjectsOrder::Ordered, {});
            visitor->AddFloatFeature(
                0,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(MED_INC))
            );
            visitor->AddFloatFeature(
                1,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(AVE_ROOMS))
            );
            visitor->AddTarget(
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(TARGET))
            );
            visitor->Finish();
        }
    );
}

TDataProviderPtr CreateFloatOnlyPool(
    const TVector<float>& firstFeature,
    const TVector<float>& secondFeature
) {
    Y_ASSERT(firstFeature.size() == secondFeature.size());
    TVector<float> target(firstFeature.size(), 0.0f);

    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                2,
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{}
            );

            visitor->Start(metaInfo, firstFeature.size(), EObjectsOrder::Ordered, {});
            visitor->AddFloatFeature(
                0,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(firstFeature))
            );
            visitor->AddFloatFeature(
                1,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(secondFeature))
            );
            visitor->AddTarget(
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(target))
            );
            visitor->Finish();
        }
    );
}

TFullModel TrainBinaryClassificationInterpolationLeakModel() {
    TDataProviders dataProviders;
    dataProviders.Learn = CreateBinaryClassificationInterpolationLeakPool();
    dataProviders.Test.push_back(dataProviders.Learn);

    NJson::TJsonValue params;
    params.InsertValue("iterations", 8);
    params.InsertValue("depth", 1);
    params.InsertValue("loss_function", "Logloss");
    params.InsertValue("learning_rate", 0.3);
    params.InsertValue("bootstrap_type", "No");
    params.InsertValue("random_strength", 0.0);
    params.InsertValue("random_seed", 0);
    params.InsertValue("thread_count", 1);
    params.InsertValue("interpolation_enabled", true);
    params.InsertValue("interpolation_type", "Linear");
    {
        NJson::TJsonValue spanModes(NJson::EJsonValueType::JSON_MAP);
        spanModes["0"] = "Absolute";
        params.InsertValue("interpolation_span_mode", std::move(spanModes));
    }
    params.InsertValue("interpolation_min_span", 0.0);
    {
        NJson::TJsonValue spans(NJson::EJsonValueType::JSON_MAP);
        spans["0"] = 0.5;
        params.InsertValue("interpolation_span", std::move(spans));
    }

    TFullModel model;
    TEvalResult evalResult;
    TrainModel(
        params,
        nullptr,
        Nothing(),
        Nothing(),
        Nothing(),
        std::move(dataProviders),
        Nothing(),
        nullptr,
        "",
        &model,
        {&evalResult}
    );
    return model;
}

TFullModel TrainRegressionInterpolationLeakModel() {
    TDataProviders dataProviders;
    dataProviders.Learn = CreateRegressionInterpolationLeakPool();
    dataProviders.Test.push_back(dataProviders.Learn);

    NJson::TJsonValue params;
    params.InsertValue("iterations", 8);
    params.InsertValue("depth", 1);
    params.InsertValue("loss_function", "RMSE");
    params.InsertValue("learning_rate", 0.3);
    params.InsertValue("bootstrap_type", "No");
    params.InsertValue("random_strength", 0.0);
    params.InsertValue("random_seed", 0);
    params.InsertValue("thread_count", 1);
    params.InsertValue("interpolation_enabled", true);
    params.InsertValue("interpolation_type", "Linear");
    {
        NJson::TJsonValue spanModes(NJson::EJsonValueType::JSON_MAP);
        spanModes["0"] = "Absolute";
        params.InsertValue("interpolation_span_mode", std::move(spanModes));
    }
    params.InsertValue("interpolation_min_span", 0.0);
    {
        NJson::TJsonValue spans(NJson::EJsonValueType::JSON_MAP);
        spans["0"] = 0.5;
        params.InsertValue("interpolation_span", std::move(spans));
    }

    TFullModel model;
    TEvalResult evalResult;
    TrainModel(
        params,
        nullptr,
        Nothing(),
        Nothing(),
        Nothing(),
        std::move(dataProviders),
        Nothing(),
        nullptr,
        "",
        &model,
        {&evalResult}
    );
    return model;
}

TFullModel BuildSingleSplitFloatModel(float border, double leftValue, double rightValue) {
    TFullModel model;
    auto* trees = model.ModelTrees.GetMutable();
    trees->SetFloatFeatures(
        {
            TFloatFeature{
                false,
                0,
                0,
                {border},
                ""
            }
        }
    );
    trees->AddBinTree({0});
    trees->AddLeafValue(leftValue);
    trees->AddLeafValue(rightValue);
    model.UpdateDynamicData();
    return model;
}

TFullModel BuildTwoSplitFloatModel(float firstBorder, float secondBorder, const TVector<double>& leafValues) {
    Y_ASSERT(leafValues.size() == 4);
    TFullModel model;
    auto* trees = model.ModelTrees.GetMutable();
    trees->SetFloatFeatures(
        {
            TFloatFeature{
                false,
                0,
                0,
                {firstBorder},
                ""
            },
            TFloatFeature{
                false,
                1,
                1,
                {secondBorder},
                ""
            }
        }
    );
    trees->AddBinTree({0, 1});
    for (double value : leafValues) {
        trees->AddLeafValue(value);
    }
    model.UpdateDynamicData();
    return model;
}

TFullModel BuildTwoTreeSingleSplitFloatModel(
    float firstBorder,
    double firstLeftValue,
    double firstRightValue,
    float secondBorder,
    double secondLeftValue,
    double secondRightValue
) {
    TFullModel model;
    auto* trees = model.ModelTrees.GetMutable();
    trees->SetFloatFeatures(
        {
            TFloatFeature{
                false,
                0,
                0,
                {firstBorder, secondBorder},
                ""
            }
        }
    );
    trees->AddBinTree({0});
    trees->AddLeafValue(firstLeftValue);
    trees->AddLeafValue(firstRightValue);
    trees->AddBinTree({1});
    trees->AddLeafValue(secondLeftValue);
    trees->AddLeafValue(secondRightValue);
    model.UpdateDynamicData();
    return model;
}

void EnableInterpolation(
    TFullModel* model,
    EFloatFeaturesInterpolationType type,
    EFloatFeaturesInterpolationSpanMode spanMode,
    double minSpan,
    const TVector<TFloatFeatureInterpolationConfig>& perFeatureConfig
) {
    TFloatFeaturesInterpolationOptions options;
    options.Enabled = true;
    options.Type = type;
    options.SpanMode = spanMode;
    options.MinSpan = minSpan;
    options.PerFloatFeatureConfig = perFeatureConfig;
    for (auto& config : options.PerFloatFeatureConfig) {
        config.SpanMode = spanMode;
        if (config.MinSpan <= 0.0) {
            config.MinSpan = minSpan;
        }
    }
    model->ModelTrees.GetMutable()->SetFloatFeaturesInterpolationOptions(std::move(options));
    model->UpdateDynamicData();
}

void CheckFlatCalcResult(
    const TFullModel& model,
    const TVector<double>& expectedPredicts,
    const TVector<ui32>& expectedLeafIndexes,
    const TVector<TConstArrayRef<float>>& features = FLOAT_FEATURES
) {
    const size_t approxDimension = model.GetDimensionsCount();
    const size_t treeCount = model.GetTreeCount();
    {
        TVector<double> predicts(features.size() * approxDimension);
        model.CalcFlat(features, predicts);
        UNIT_ASSERT_EQUAL(expectedPredicts, predicts);

        TVector<ui32> leafIndexes(features.size() * treeCount, 100);
        model.CalcLeafIndexes(features, {}, leafIndexes);
        UNIT_ASSERT_EQUAL(expectedLeafIndexes, leafIndexes);
    }

    for (size_t sampleIndex = 0; sampleIndex < features.size(); ++sampleIndex) {
        const auto sampleFeatures = features[sampleIndex];
        const TVector<double> expectedSamplePredict(
            expectedPredicts.begin() + sampleIndex * approxDimension,
            expectedPredicts.begin() + (sampleIndex + 1) * approxDimension
        );
        TVector<double> samplePredict(model.GetDimensionsCount());
        model.CalcFlatSingle(sampleFeatures, samplePredict);
        UNIT_ASSERT_EQUAL(expectedSamplePredict, samplePredict);

        const TVector<TCalcerIndexType> expectedSampleIndexes(
            expectedLeafIndexes.begin() + sampleIndex * treeCount,
            expectedLeafIndexes.begin() + (sampleIndex + 1) * treeCount
        );
        TVector<TCalcerIndexType> sampleLeafIndexes(treeCount);
        model.CalcLeafIndexesSingle(sampleFeatures, {}, sampleLeafIndexes);
        UNIT_ASSERT_EQUAL(expectedSampleIndexes, sampleLeafIndexes);
    }
}

bool IsGpuEvaluatorSupported() {
    for (auto evaluatorType : TFullModel::GetSupportedEvaluatorTypes()) {
        if (evaluatorType == EFormulaEvaluatorType::GPU) {
            return true;
        }
    }
    return false;
}

void CheckFlatCalcPredictionsOnly(
    const TFullModel& model,
    const TVector<double>& expectedPredicts,
    const TVector<TConstArrayRef<float>>& features
) {
    TVector<double> predicts(features.size() * model.GetDimensionsCount());
    model.CalcFlat(features, predicts);
    UNIT_ASSERT_EQUAL(expectedPredicts.size(), predicts.size());
    for (size_t i = 0; i < predicts.size(); ++i) {
        UNIT_ASSERT_DOUBLES_EQUAL(expectedPredicts[i], predicts[i], 1e-6);
    }
}

Y_UNIT_TEST_SUITE(TObliviousTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto model = SimpleFloatModel();
        CheckFlatCalcResult(model, xrange<double>(8), xrange<ui32>(8));
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, xrange<double>(8), xrange<ui32>(8));
    }

    Y_UNIT_TEST(TestFlatCalcFloatWithScaleAndBias) {
        auto model = SimpleFloatModel();
        model.SetScaleAndBias({0.5, {0.125}});
        auto norm = model.GetScaleAndBias();
        TVector<double> expectedPredicts;
        double bias = norm.GetOneDimensionalBias();
        for (int sampleId : xrange(8)) {
            expectedPredicts.push_back(sampleId * norm.Scale + bias);
        }
        CheckFlatCalcResult(model, expectedPredicts, xrange<ui32>(8));
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, xrange<ui32>(8));
    }

    Y_UNIT_TEST(TestTwoTrees) {
        auto model = SimpleFloatModel(2);
        TVector<ui32> expectedLeafIndexes;
        TVector<double> expectedPredicts;
        for (ui32 sampleId = 0; sampleId < 8; ++sampleId) {
            expectedLeafIndexes.push_back(sampleId);
            expectedLeafIndexes.push_back(sampleId);
            expectedPredicts.push_back(11.0 * sampleId);
        }
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes);
    }

    Y_UNIT_TEST(TestFlatCalcOnDeepTree) {
        const size_t treeDepth = 9;
        auto model = SimpleDeepTreeModel(treeDepth);

        TVector<TVector<float>> data;
        TVector<TCalcerIndexType> expectedLeafIndexes;
        TVector<double> expectedPredicts;
        for (size_t sampleId : xrange(1 << treeDepth)) {
            expectedLeafIndexes.push_back(sampleId);
            expectedPredicts.push_back(sampleId);
            TVector<float> sampleFeatures(treeDepth);
            for (auto featureId : xrange(treeDepth)) {
                sampleFeatures[featureId] = sampleId % 2;
                sampleId = sampleId >> 1;
            }
            data.push_back(std::move(sampleFeatures));
        }
        const auto features = GetFeatureRef(data);
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);
    }

    Y_UNIT_TEST(TestFlatCalcMultiVal) {
        auto model = MultiValueFloatModel();
        TVector<TConstArrayRef<float>> features(FLOAT_FEATURES.begin(), FLOAT_FEATURES.begin() + 4);
        TVector<double> expectedPredicts = {
            00., 10., 20.,
            01., 11., 21.,
            02., 12., 22.,
            03., 13., 23.,
        };
        CheckFlatCalcResult(model, expectedPredicts, xrange(4), features);
        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        CheckFlatCalcResult(model, expectedPredicts, xrange(4), features);
    }

    Y_UNIT_TEST(TestFlatCalcMultiValMultiProba) {
        auto model = MultiValueFloatModel();
        constexpr size_t DocCount = 4;
        TConstArrayRef<TConstArrayRef<float>> features(FLOAT_FEATURES.begin(), DocCount);
        TVector<double> expectedProbs = {
            Sigmoid(00.), Sigmoid(10.), Sigmoid(20.),
            Sigmoid(01.), Sigmoid(11.), Sigmoid(21.),
            Sigmoid(02.), Sigmoid(12.), Sigmoid(22.),
            Sigmoid(03.), Sigmoid(13.), Sigmoid(23.),
        };
        auto customEval = model.GetCurrentEvaluator()->Clone();
        customEval->SetPredictionType(NCB::NModelEvaluation::EPredictionType::MultiProbability);
        TVector<double> probs(model.GetDimensionsCount() * DocCount, 0);
        customEval->Calc<TStringBuf>(features, {}, probs);
        for (auto i : xrange(model.GetDimensionsCount() * DocCount)) {
            UNIT_ASSERT_DOUBLES_EQUAL(expectedProbs[i], probs[i], 1.0e-6);
        }
    }

    Y_UNIT_TEST(TestSingleSplitLinearInterpolationAbsoluteSpan) {
        auto model = BuildSingleSplitFloatModel(10.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Absolute,
            0.0,
            {{0, 2.0}}
        );

        TVector<TVector<float>> data = {{8.f}, {9.f}, {10.f}, {11.f}, {12.f}};
        TVector<double> expectedPredicts = {0.0, 2.5, 5.0, 7.5, 10.0};
        TVector<ui32> expectedLeafIndexes = {0, 0, 0, 1, 1};
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, GetFeatureRef(data));
    }

    Y_UNIT_TEST(TestSingleSplitSigmoidInterpolationAbsoluteSpan) {
        auto model = BuildSingleSplitFloatModel(10.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Sigmoid,
            EFloatFeaturesInterpolationSpanMode::Absolute,
            0.0,
            {{0, 2.0}}
        );

        TVector<TVector<float>> data = {{8.f}, {10.f}, {12.f}};
        TVector<double> expectedPredicts = {
            10.0 * Sigmoid(-5.0),
            5.0,
            10.0 * Sigmoid(5.0)
        };
        TVector<ui32> expectedLeafIndexes = {0, 0, 1};
        const auto features = GetFeatureRef(data);

        TVector<double> predicts(features.size());
        model.CalcFlat(features, predicts);
        for (size_t i = 0; i < predicts.size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL(expectedPredicts[i], predicts[i], 1e-8);
        }
        TVector<ui32> leafIndexes(features.size());
        model.CalcLeafIndexes(features, {}, leafIndexes);
        UNIT_ASSERT_EQUAL(expectedLeafIndexes, leafIndexes);
    }

    Y_UNIT_TEST(TestTwoSplitMixedInterpolation) {
        auto model = BuildTwoSplitFloatModel(10.0f, 20.0f, {0.0, 10.0, 100.0, 110.0});
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Absolute,
            0.0,
            {{0, 2.0}}
        );

        TVector<TVector<float>> data = {{9.f, 25.f}, {10.f, 25.f}, {11.f, 25.f}};
        TVector<double> expectedPredicts = {102.5, 105.0, 107.5};
        TVector<ui32> expectedLeafIndexes = {2, 2, 3};
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, GetFeatureRef(data));
    }

    Y_UNIT_TEST(TestSingleSplitLinearInterpolationRelativeSpanAndSerialization) {
        auto model = BuildSingleSplitFloatModel(10.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Relative,
            0.0,
            {{0, 0.1}}
        );

        TVector<TVector<float>> data = {{9.5f}, {10.0f}, {10.5f}};
        TVector<double> expectedPredicts = {2.5, 5.0, 7.5};
        TVector<ui32> expectedLeafIndexes = {0, 0, 1};
        const auto features = GetFeatureRef(data);
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);

        TStringStream stream;
        model.Save(&stream);
        TFullModel loadedModel;
        loadedModel.Load(&stream);
        CheckFlatCalcResult(loadedModel, expectedPredicts, expectedLeafIndexes, features);
    }

    Y_UNIT_TEST(TestSingleSplitLinearInterpolationRelativeSpanUsesMinSpan) {
        auto model = BuildSingleSplitFloatModel(1.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Relative,
            2.0,
            {{0, 0.1}}
        );

        TVector<TVector<float>> data = {{-1.f}, {0.f}, {1.f}, {2.f}, {3.f}};
        TVector<double> expectedPredicts = {0.0, 2.5, 5.0, 7.5, 10.0};
        TVector<ui32> expectedLeafIndexes = {0, 0, 0, 1, 1};
        const auto features = GetFeatureRef(data);
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);

        TStringStream stream;
        model.Save(&stream);
        TFullModel loadedModel;
        loadedModel.Load(&stream);
        CheckFlatCalcResult(loadedModel, expectedPredicts, expectedLeafIndexes, features);
    }

    Y_UNIT_TEST(TestSingleSplitLinearInterpolationRelativeSpanUsesPerFeatureMinSpan) {
        auto model = BuildSingleSplitFloatModel(1.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Relative,
            0.0,
            {{0, 0.1, EFloatFeaturesInterpolationSpanMode::Relative, 2.0}}
        );

        TVector<TVector<float>> data = {{-1.f}, {0.f}, {1.f}, {2.f}, {3.f}};
        TVector<double> expectedPredicts = {0.0, 2.5, 5.0, 7.5, 10.0};
        TVector<ui32> expectedLeafIndexes = {0, 0, 0, 1, 1};
        const auto features = GetFeatureRef(data);
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);

        TStringStream stream;
        model.Save(&stream);
        TFullModel loadedModel;
        loadedModel.Load(&stream);
        CheckFlatCalcResult(loadedModel, expectedPredicts, expectedLeafIndexes, features);
    }

    Y_UNIT_TEST(TestTwoSplitPerFeatureSpanModes) {
        auto model = BuildTwoSplitFloatModel(10.0f, 20.0f, {0.0, 10.0, 100.0, 110.0});
        TFloatFeaturesInterpolationOptions options;
        options.Enabled = true;
        options.Type = EFloatFeaturesInterpolationType::Linear;
        options.MinSpan = 0.0;
        options.PerFloatFeatureConfig = {
            {0, 2.0, EFloatFeaturesInterpolationSpanMode::Absolute, 0.0},
            {1, 0.1, EFloatFeaturesInterpolationSpanMode::Relative, 0.0}
        };
        model.ModelTrees.GetMutable()->SetFloatFeaturesInterpolationOptions(std::move(options));
        model.UpdateDynamicData();

        TVector<TVector<float>> data = {{11.f, 21.f}};
        TVector<double> expectedPredicts = {82.5};
        TVector<ui32> expectedLeafIndexes = {3};
        const auto features = GetFeatureRef(data);
        CheckFlatCalcResult(model, expectedPredicts, expectedLeafIndexes, features);

        TStringStream stream;
        model.Save(&stream);
        TFullModel loadedModel;
        loadedModel.Load(&stream);
        CheckFlatCalcResult(loadedModel, expectedPredicts, expectedLeafIndexes, features);
    }

    Y_UNIT_TEST(TestSingleSplitLinearInterpolationAbsoluteSpanGpu) {
        if (!IsGpuEvaluatorSupported()) {
            return;
        }

        auto model = BuildSingleSplitFloatModel(10.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Absolute,
            0.0,
            {{0, 2.0}}
        );
        model.SetEvaluatorType(EFormulaEvaluatorType::GPU);

        TVector<TVector<float>> data = {{8.f}, {9.f}, {10.f}, {11.f}, {12.f}};
        TVector<double> expectedPredicts = {0.0, 2.5, 5.0, 7.5, 10.0};
        CheckFlatCalcPredictionsOnly(model, expectedPredicts, GetFeatureRef(data));
    }

    Y_UNIT_TEST(TestSingleSplitLinearInterpolationRelativeSpanUsesMinSpanGpu) {
        if (!IsGpuEvaluatorSupported()) {
            return;
        }

        auto model = BuildSingleSplitFloatModel(1.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Relative,
            2.0,
            {{0, 0.1}}
        );
        model.SetEvaluatorType(EFormulaEvaluatorType::GPU);

        TVector<TVector<float>> data = {{-1.f}, {0.f}, {1.f}, {2.f}, {3.f}};
        TVector<double> expectedPredicts = {0.0, 2.5, 5.0, 7.5, 10.0};
        CheckFlatCalcPredictionsOnly(model, expectedPredicts, GetFeatureRef(data));
    }

    Y_UNIT_TEST(TestSingleSplitSigmoidInterpolationAbsoluteSpanGpu) {
        if (!IsGpuEvaluatorSupported()) {
            return;
        }

        auto model = BuildSingleSplitFloatModel(10.0f, 0.0, 10.0);
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Sigmoid,
            EFloatFeaturesInterpolationSpanMode::Absolute,
            0.0,
            {{0, 2.0}}
        );
        model.SetEvaluatorType(EFormulaEvaluatorType::GPU);

        TVector<TVector<float>> data = {{8.f}, {10.f}, {12.f}};
        TVector<double> expectedPredicts = {
            10.0 * Sigmoid(-5.0),
            5.0,
            10.0 * Sigmoid(5.0)
        };
        CheckFlatCalcPredictionsOnly(model, expectedPredicts, GetFeatureRef(data));
    }

    Y_UNIT_TEST(TestInterpolationIsTreeLocalForMultipleTrees) {
        auto model = BuildTwoTreeSingleSplitFloatModel(
            10.0f,
            0.0,
            10.0,
            20.0f,
            100.0,
            200.0
        );
        EnableInterpolation(
            &model,
            EFloatFeaturesInterpolationType::Linear,
            EFloatFeaturesInterpolationSpanMode::Absolute,
            0.0,
            {{0, 2.0}}
        );

        TVector<TVector<float>> data = {{10.f}, {15.f}, {20.f}};
        TVector<double> expectedPredicts = {
            5.0 + 100.0,
            10.0 + 100.0,
            10.0 + 150.0
        };
        CheckFlatCalcPredictionsOnly(model, expectedPredicts, GetFeatureRef(data));
    }

    Y_UNIT_TEST(TestBinaryClassificationInterpolationTreeLocality) {
        const auto model = TrainBinaryClassificationInterpolationLeakModel();
        auto hardModel = model;
        hardModel.ModelTrees.GetMutable()->SetFloatFeaturesInterpolationOptions({});
        hardModel.UpdateDynamicData();
        const auto& trees = *model.ModelTrees;
        const auto& treeSizes = trees.GetModelTreeData()->GetTreeSizes();
        const auto& treeStartOffsets = trees.GetModelTreeData()->GetTreeStartOffsets();
        const auto& treeSplits = trees.GetModelTreeData()->GetTreeSplits();
        const auto& binFeatures = trees.GetBinFeatures();
        const auto& floatFeatures = trees.GetFloatFeatures();
        const auto& applyData = *trees.GetApplyData();
        const size_t treeCount = model.GetTreeCount();

        UNIT_ASSERT_VALUES_EQUAL(floatFeatures.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(floatFeatures[0].Position.Index, 0);
        UNIT_ASSERT_VALUES_EQUAL(floatFeatures[0].Position.FlatIndex, 0);
        UNIT_ASSERT_VALUES_EQUAL(floatFeatures[1].Position.Index, 1);
        UNIT_ASSERT_VALUES_EQUAL(floatFeatures[1].Position.FlatIndex, 1);
        UNIT_ASSERT_VALUES_EQUAL(applyData.FloatFeatureInterpolationSpans.size(), 2);
        UNIT_ASSERT_DOUBLES_EQUAL(applyData.FloatFeatureInterpolationSpans[0], 0.5, 1e-12);
        UNIT_ASSERT_DOUBLES_EQUAL(applyData.FloatFeatureInterpolationSpans[1], -1.0, 1e-12);

        bool hasNonInterpolatedTree = false;
        for (size_t treeId = 0; treeId < treeCount; ++treeId) {
            UNIT_ASSERT_VALUES_EQUAL(treeSizes[treeId], 1);
            const auto& split = binFeatures[treeSplits[treeStartOffsets[treeId]]];
            if (split.Type == ESplitType::FloatFeature && split.FloatFeature.FloatFeature != 0) {
                hasNonInterpolatedTree = true;
            }
        }
        UNIT_ASSERT(hasNonInterpolatedTree);

        TVector<float> sample = {25.38f, 0.3001f};
        TVector<double> treePredictions(treeCount);
        TVector<double> hardTreePredictions(treeCount);
        for (size_t treeId = 0; treeId < treeCount; ++treeId) {
            TVector<double> result(1);
            model.CalcFlatSingle(sample, treeId, treeId + 1, result);
            treePredictions[treeId] = result[0];
            hardModel.CalcFlatSingle(sample, treeId, treeId + 1, result);
            hardTreePredictions[treeId] = result[0];
        }

        for (size_t pointId = 0; pointId < 25; ++pointId) {
            sample[0] = 10.534f + (25.38f - 10.534f) * pointId / 24.0f;
            for (size_t treeId = 0; treeId < treeCount; ++treeId) {
                const auto& split = binFeatures[treeSplits[treeStartOffsets[treeId]]];
                if (split.Type != ESplitType::FloatFeature || split.FloatFeature.FloatFeature == 0) {
                    continue;
                }
                TVector<double> result(1);
                model.CalcFlatSingle(sample, treeId, treeId + 1, result);
                TVector<double> hardResult(1);
                hardModel.CalcFlatSingle(sample, treeId, treeId + 1, hardResult);
                UNIT_ASSERT_DOUBLES_EQUAL_C(
                    hardResult[0],
                    hardTreePredictions[treeId],
                    1e-9,
                    "hard model changed"
                    << " treeId=" << treeId
                    << " splitFeature=" << split.FloatFeature.FloatFeature
                    << " pointId=" << pointId
                    << " sample0=" << sample[0]
                );
                UNIT_ASSERT_DOUBLES_EQUAL_C(
                    result[0],
                    treePredictions[treeId],
                    1e-9,
                    "interpolated model changed"
                    << " treeId=" << treeId
                    << " splitFeature=" << split.FloatFeature.FloatFeature
                    << " pointId=" << pointId
                    << " sample0=" << sample[0]
                    << " interpolated=" << result[0]
                    << " baseline=" << treePredictions[treeId]
                );
            }
        }
    }

    Y_UNIT_TEST(TestRegressionInterpolationTreeLocality) {
        const auto model = TrainRegressionInterpolationLeakModel();
        const auto& trees = *model.ModelTrees;
        const auto& treeSizes = trees.GetModelTreeData()->GetTreeSizes();
        const auto& treeStartOffsets = trees.GetModelTreeData()->GetTreeStartOffsets();
        const auto& treeSplits = trees.GetModelTreeData()->GetTreeSplits();
        const auto& binFeatures = trees.GetBinFeatures();
        const size_t treeCount = model.GetTreeCount();

        bool hasNonInterpolatedTree = false;
        for (size_t treeId = 0; treeId < treeCount; ++treeId) {
            UNIT_ASSERT_VALUES_EQUAL(treeSizes[treeId], 1);
            const auto& split = binFeatures[treeSplits[treeStartOffsets[treeId]]];
            if (split.Type == ESplitType::FloatFeature && split.FloatFeature.FloatFeature != 0) {
                hasNonInterpolatedTree = true;
            }
        }
        UNIT_ASSERT(hasNonInterpolatedTree);

        TVector<float> sample = {8.3252f, 6.984126984126984f};
        TVector<double> treePredictions(treeCount);
        for (size_t treeId = 0; treeId < treeCount; ++treeId) {
            TVector<double> result(1);
            model.CalcFlatSingle(sample, treeId, treeId + 1, result);
            treePredictions[treeId] = result[0];
        }

        for (size_t pointId = 0; pointId < 25; ++pointId) {
            sample[0] = 1.3578f + (8.3252f - 1.3578f) * pointId / 24.0f;
            for (size_t treeId = 0; treeId < treeCount; ++treeId) {
                const auto& split = binFeatures[treeSplits[treeStartOffsets[treeId]]];
                if (split.Type != ESplitType::FloatFeature || split.FloatFeature.FloatFeature == 0) {
                    continue;
                }
                TVector<double> result(1);
                model.CalcFlatSingle(sample, treeId, treeId + 1, result);
                UNIT_ASSERT_DOUBLES_EQUAL_C(
                    result[0],
                    treePredictions[treeId],
                    1e-9,
                    "interpolated regression model changed"
                    << " treeId=" << treeId
                    << " splitFeature=" << split.FloatFeature.FloatFeature
                    << " pointId=" << pointId
                    << " sample0=" << sample[0]
                    << " interpolated=" << result[0]
                    << " baseline=" << treePredictions[treeId]
                );
            }
        }
    }

    Y_UNIT_TEST(TestQuantizedInterpolationDoesNotLeakAcrossEvaluationBlocks) {
        const TFullModel model = TrainRegressionInterpolationLeakModel();
        const auto& trees = *model.ModelTrees;

        TMaybe<float> interpolatedBorder;
        for (int splitIdx : trees.GetModelTreeData()->GetTreeSplits()) {
            const auto& split = trees.GetBinFeatures()[splitIdx];
            if (split.Type == ESplitType::FloatFeature && split.FloatFeature.FloatFeature == 0) {
                interpolatedBorder = split.FloatFeature.Split;
                break;
            }
        }
        UNIT_ASSERT(interpolatedBorder.Defined());

        TVector<float> baseFirstFeature(160, *interpolatedBorder);
        TVector<float> shiftedFirstFeature = baseFirstFeature;
        for (size_t docId = 0; docId < 128; ++docId) {
            shiftedFirstFeature[docId] = *interpolatedBorder - 10.0f;
        }

        TVector<float> secondFeature(160);
        for (size_t docId = 0; docId < secondFeature.size(); ++docId) {
            secondFeature[docId] = static_cast<float>(docId % 7);
        }

        const auto basePool = CreateFloatOnlyPool(baseFirstFeature, secondFeature);
        const auto shiftedPool = CreateFloatOnlyPool(shiftedFirstFeature, secondFeature);

        const auto baseQuantized = MakeQuantizedFeaturesForEvaluator(model, *basePool->ObjectsData);
        const auto shiftedQuantized = MakeQuantizedFeaturesForEvaluator(model, *shiftedPool->ObjectsData);

        TVector<double> basePredictions(160);
        TVector<double> shiftedPredictions(160);

        model.GetCurrentEvaluator()->Calc(
            baseQuantized.Get(),
            0,
            model.GetTreeCount(),
            basePredictions
        );
        model.GetCurrentEvaluator()->Calc(
            shiftedQuantized.Get(),
            0,
            model.GetTreeCount(),
            shiftedPredictions
        );

        for (size_t docId = 128; docId < 160; ++docId) {
            UNIT_ASSERT_DOUBLES_EQUAL(basePredictions[docId], shiftedPredictions[docId], 1e-12);
        }
    }

    Y_UNIT_TEST(TestQuantizedInterpolationDoesNotLeakAcrossEvaluationBlocksGpu) {
        if (!IsGpuEvaluatorSupported()) {
            return;
        }

        TFullModel model = TrainRegressionInterpolationLeakModel();
        model.SetEvaluatorType(EFormulaEvaluatorType::GPU);
        const auto& trees = *model.ModelTrees;

        TMaybe<float> interpolatedBorder;
        for (int splitIdx : trees.GetModelTreeData()->GetTreeSplits()) {
            const auto& split = trees.GetBinFeatures()[splitIdx];
            if (split.Type == ESplitType::FloatFeature && split.FloatFeature.FloatFeature == 0) {
                interpolatedBorder = split.FloatFeature.Split;
                break;
            }
        }
        UNIT_ASSERT(interpolatedBorder.Defined());

        TVector<float> baseFirstFeature(160, *interpolatedBorder);
        TVector<float> shiftedFirstFeature = baseFirstFeature;
        for (size_t docId = 0; docId < 128; ++docId) {
            shiftedFirstFeature[docId] = *interpolatedBorder - 10.0f;
        }

        TVector<float> secondFeature(160);
        for (size_t docId = 0; docId < secondFeature.size(); ++docId) {
            secondFeature[docId] = static_cast<float>(docId % 7);
        }

        const auto basePool = CreateFloatOnlyPool(baseFirstFeature, secondFeature);
        const auto shiftedPool = CreateFloatOnlyPool(shiftedFirstFeature, secondFeature);

        const auto baseQuantized = MakeQuantizedFeaturesForEvaluator(model, *basePool->ObjectsData);
        const auto shiftedQuantized = MakeQuantizedFeaturesForEvaluator(model, *shiftedPool->ObjectsData);

        TVector<double> basePredictions(160);
        TVector<double> shiftedPredictions(160);

        model.GetCurrentEvaluator()->Calc(
            baseQuantized.Get(),
            0,
            model.GetTreeCount(),
            basePredictions
        );
        model.GetCurrentEvaluator()->Calc(
            shiftedQuantized.Get(),
            0,
            model.GetTreeCount(),
            shiftedPredictions
        );

        for (size_t docId = 128; docId < 160; ++docId) {
            UNIT_ASSERT_DOUBLES_EQUAL(basePredictions[docId], shiftedPredictions[docId], 1e-12);
        }
    }

    Y_UNIT_TEST(TestCatOnlyModel) {
        const auto model = TrainCatOnlyModel();

        const auto applySingle = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}};
            double result = 0.;
            model.Calc({}, f, MakeArrayRef(&result, 1));
        };
        UNIT_ASSERT_NO_EXCEPTION(applySingle());

        const auto applyBatch = [&] {
            const TVector<TStringBuf> f[] = {{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "k"}};
            double results[3];
            model.Calc({}, f, results);
        };
        UNIT_ASSERT_NO_EXCEPTION(applyBatch());
    }

    static void CheckCalcTextResult(
        const TFullModel& model,
        TConstArrayRef<TVector<TStringBuf>> transposedTextFeatures,
        TConstArrayRef<double> expectedResults,
        const ui32 docCount,
        const ui32 numEstimatedFeatures
    ) {
        TVector<double> results(docCount);
        const double epsilon = 1e-8;

        for (ui32 estimatedFeatureId = 0; estimatedFeatureId < numEstimatedFeatures; estimatedFeatureId++) {
            ui32 treeIndex = estimatedFeatureId;
            model.Calc(
                {},
                TConstArrayRef<TVector<TStringBuf>>{},
                transposedTextFeatures,
                treeIndex,
                treeIndex + 1,
                MakeArrayRef(results)
            );

            for (ui32 docId: xrange(docCount)) {
                UNIT_ASSERT_DOUBLES_EQUAL(
                    expectedResults[estimatedFeatureId * docCount + docId],
                    results[docId],
                    epsilon
                );
            }
        }
    }

    Y_UNIT_TEST(TestTextOnlyModel) {
        TVector<NCBTest::TTextFeature> features;
        TMap<ui32, NCBTest::TTokenizedTextFeature> tokenizedFeatures;
        TVector<TDigitizer> digitizers;
        TVector<TTextFeatureCalcerPtr> calcers;
        TVector<TVector<ui32>> perFeatureDigitizers;
        TVector<TVector<ui32>> perTokenizedFeatureCalcers;

        NCBTest::CreateTextDataForTest(
            &features,
            &tokenizedFeatures,
            &digitizers,
            &calcers,
            &perFeatureDigitizers,
            &perTokenizedFeatureCalcers
        );

        auto textProcessingCollection = MakeIntrusive<TTextProcessingCollection>(
            digitizers,
            calcers,
            perFeatureDigitizers,
            perTokenizedFeatureCalcers
        );

        const ui32 docCount = features[0].size();
        const ui32 numTextFeatures = features.size();
        const ui32 numEstimatedFeatures = textProcessingCollection->TotalNumberOfOutputFeatures();

        TVector<TVector<TStringBuf>> textFeatures;
        for (auto& feature: features) {
            textFeatures.emplace_back(feature.begin(), feature.end());
        }

        TVector<double> expectedResults(numEstimatedFeatures * docCount);
        auto model = SimpleTextModel(
            textProcessingCollection,
            MakeConstArrayRef(textFeatures),
            MakeArrayRef(expectedResults)
        );

        TVector<TVector<TStringBuf>> transposedTextFeatures;
        for (ui32 docId: xrange(docCount)) {
            auto& ref = transposedTextFeatures.emplace_back();
            for (ui32 featureId: xrange(numTextFeatures)) {
                ref.emplace_back(features[featureId][docId]);
            }
        }

        CheckCalcTextResult(
            model,
            MakeConstArrayRef(transposedTextFeatures),
            MakeConstArrayRef(expectedResults),
            docCount,
            numEstimatedFeatures
        );

        model.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();

        CheckCalcTextResult(
            model,
            MakeConstArrayRef(transposedTextFeatures),
            MakeConstArrayRef(expectedResults),
            docCount,
            numEstimatedFeatures
        );
    }
}

Y_UNIT_TEST_SUITE(TNonSymmetricTreeModel) {
    Y_UNIT_TEST(TestFlatCalcFloat) {
        auto modelCalcer = SimpleAsymmetricModel();
        TVector<double> canonVals = {
            101., 203., 102., 303.,
            111., 213., 112., 313.};
        TVector<ui32> expectedLeafIndexes = {
            1, 0, 0,
            0, 0, 1,
            2, 0, 0,
            0, 0, 2,
            1, 1, 0,
            0, 1, 1,
            2, 1, 0,
            0, 1, 2
        };
        CheckFlatCalcResult(modelCalcer, canonVals, expectedLeafIndexes);

        TStringStream strStream;
        modelCalcer.Save(&strStream);
        TFullModel deserializedModel;
        deserializedModel.Load(&strStream);
        CheckFlatCalcResult(deserializedModel, canonVals, expectedLeafIndexes);
    }
}
