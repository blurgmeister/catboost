#include <catboost/libs/model/ut/lib/model_test_helpers.h>
#include <catboost/libs/model/features.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/model/model_export/json_model_helpers.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/learn_context.h>

#include <library/cpp/json/json_writer.h>

#include <library/cpp/testing/unittest/registar.h>

#include <algorithm>
#include <cmath>

using namespace std;
using namespace NCB;

void DoSerializeDeserialize(const TFullModel& model) {
    TStringStream strStream;
    model.Save(&strStream);
    TFullModel deserializedModel;
    deserializedModel.Load(&strStream);
    UNIT_ASSERT_EQUAL(model, deserializedModel);
}

static TDataProviderPtr CreateInterpolationMappingPool() {
    constexpr ui32 FeatureCount = 4;
    constexpr ui32 DocCount = 8;

    TVector<TVector<float>> floatFeatures = {
        TVector<float>(DocCount, 0.0f),
        {0.f, 0.f, 0.f, 0.f, 10.f, 10.f, 10.f, 10.f},
        TVector<float>(DocCount, 1.0f)
    };
    TVector<TStringBuf> catFeature(DocCount, "a");
    TVector<float> target = {0.f, 0.f, 0.f, 0.f, 10.f, 10.f, 10.f, 10.f};

    return CreateDataProvider(
        [&] (IRawFeaturesOrderDataVisitor* visitor) {
            TDataMetaInfo metaInfo;
            metaInfo.TargetType = ERawTargetType::Float;
            metaInfo.TargetCount = 1;
            metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                FeatureCount,
                TVector<ui32>{1},
                TVector<ui32>{},
                TVector<ui32>{},
                TVector<TString>{}
            );

            visitor->Start(metaInfo, DocCount, EObjectsOrder::Ordered, {});
            visitor->AddFloatFeature(
                0,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(floatFeatures[0]))
            );
            visitor->AddCatFeature(1, catFeature);
            visitor->AddFloatFeature(
                2,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(floatFeatures[1]))
            );
            visitor->AddFloatFeature(
                3,
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(TVector<float>(floatFeatures[2]))
            );
            visitor->AddTarget(
                MakeIntrusive<TTypeCastArrayHolder<float, float>>(std::move(target))
            );
            visitor->Finish();
        }
    );
}

Y_UNIT_TEST_SUITE(TModelSerialization) {
    Y_UNIT_TEST(TestSerializeDeserializeFullModel) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        DoSerializeDeserialize(trainedModel);
        trainedModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        DoSerializeDeserialize(trainedModel);
    }

    Y_UNIT_TEST(TestSerializeDeserializeFullModelWithScaleAndBias) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        trainedModel.SetScaleAndBias({0.5, {0.125}});
        DoSerializeDeserialize(trainedModel);
        trainedModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        DoSerializeDeserialize(trainedModel);
    }

    Y_UNIT_TEST(TestSerializeDeserializeInterpolationUsesInternalFloatFeatureIndexMapping) {
        TDataProviders dataProviders;
        dataProviders.Learn = CreateInterpolationMappingPool();
        dataProviders.Test.push_back(dataProviders.Learn);

        NJson::TJsonValue params;
        params.InsertValue("iterations", 1);
        params.InsertValue("depth", 1);
        params.InsertValue("loss_function", "RMSE");
        params.InsertValue("learning_rate", 1.0);
        params.InsertValue("bootstrap_type", "No");
        params.InsertValue("random_strength", 0.0);
        params.InsertValue("random_seed", 0);
        params.InsertValue("interpolation_enabled", true);
        params.InsertValue("interpolation_type", "Linear");
        params.InsertValue("interpolation_span_mode", "Absolute");
        params.InsertValue("interpolation_min_span", 0.0);
        {
            NJson::TJsonValue ignoredFeatures(NJson::EJsonValueType::JSON_ARRAY);
            ignoredFeatures.AppendValue(1);
            params.InsertValue("ignored_features", std::move(ignoredFeatures));
        }
        {
            NJson::TJsonValue spans(NJson::EJsonValueType::JSON_MAP);
            spans["2"] = 1.0;
            params.InsertValue("interpolation_span_per_float_feature", std::move(spans));
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

        const auto& floatFeatures = model.ModelTrees->GetFloatFeatures();
        const auto mappedFeatureIt = std::find_if(floatFeatures.begin(), floatFeatures.end(), [] (const TFloatFeature& feature) {
            return feature.Position.FlatIndex == 2;
        });
        UNIT_ASSERT(mappedFeatureIt != floatFeatures.end());
        UNIT_ASSERT_VALUES_EQUAL(mappedFeatureIt->Position.Index, 1);

        const auto& interpolationOptions = model.ModelTrees->GetFloatFeaturesInterpolationOptions();
        UNIT_ASSERT(interpolationOptions.Enabled);
        UNIT_ASSERT_VALUES_EQUAL(interpolationOptions.PerFloatFeatureConfig.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(interpolationOptions.PerFloatFeatureConfig[0].FloatFeatureIndex, 1);
        UNIT_ASSERT_DOUBLES_EQUAL(interpolationOptions.PerFloatFeatureConfig[0].Span, 1.0, 1e-12);

        TStringStream stream;
        model.Save(&stream);

        TFullModel loadedModel;
        loadedModel.Load(&stream);

        const auto& loadedOptions = loadedModel.ModelTrees->GetFloatFeaturesInterpolationOptions();
        UNIT_ASSERT(loadedOptions.Enabled);
        UNIT_ASSERT_VALUES_EQUAL(loadedOptions.PerFloatFeatureConfig.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(loadedOptions.PerFloatFeatureConfig[0].FloatFeatureIndex, 1);
        UNIT_ASSERT_DOUBLES_EQUAL(loadedOptions.PerFloatFeatureConfig[0].Span, 1.0, 1e-12);

        const auto loadedFeatureIt = std::find_if(
            loadedModel.ModelTrees->GetFloatFeatures().begin(),
            loadedModel.ModelTrees->GetFloatFeatures().end(),
            [] (const TFloatFeature& feature) {
                return feature.Position.FlatIndex == 2;
            }
        );
        UNIT_ASSERT(loadedFeatureIt != loadedModel.ModelTrees->GetFloatFeatures().end());
        const float border = loadedFeatureIt->Borders[0];

        TVector<TVector<float>> docStorage = {
            {0.f, 0.f, border - 2.0f, 1.0f},
            {0.f, 0.f, border, 1.0f},
            {0.f, 0.f, border + 2.0f, 1.0f}
        };
        TVector<TConstArrayRef<float>> features(docStorage.size());
        for (size_t i = 0; i < docStorage.size(); ++i) {
            features[i] = docStorage[i];
        }

        TVector<double> interpolatedPredictions(features.size());
        loadedModel.CalcFlat(features, interpolatedPredictions);

        TFullModel hardModel = loadedModel;
        hardModel.ModelTrees.GetMutable()->SetFloatFeaturesInterpolationOptions({});
        hardModel.UpdateDynamicData();

        TVector<double> hardPredictions(features.size());
        hardModel.CalcFlat(features, hardPredictions);

        UNIT_ASSERT_DOUBLES_EQUAL(interpolatedPredictions[0], hardPredictions[0], 1e-9);
        UNIT_ASSERT_DOUBLES_EQUAL(interpolatedPredictions[2], hardPredictions[2], 1e-9);
        UNIT_ASSERT(interpolatedPredictions[1] > Min(hardPredictions[0], hardPredictions[2]));
        UNIT_ASSERT(interpolatedPredictions[1] < Max(hardPredictions[0], hardPredictions[2]));
        UNIT_ASSERT(std::abs(interpolatedPredictions[1] - hardPredictions[1]) > 1e-9);
    }

    Y_UNIT_TEST(TestSerializeDeserializeFullModelNonOwning) {
        auto check = [&](const TFullModel& model) {
            TStringStream strStream;
            model.Save(&strStream);
            TFullModel deserializedModel;
            deserializedModel.InitNonOwning(strStream.Data(), strStream.Size());
            UNIT_ASSERT_EQUAL(model, deserializedModel);
        };
        check(TrainFloatCatboostModel());
        check(TrainCatOnlyNoOneHotModel());
    }

    Y_UNIT_TEST(TestSerializeDeserializeCoreML) {
        TFullModel trainedModel = TrainFloatCatboostModel();
        TStringStream strStream;
        trainedModel.Save(&strStream);
        ExportModel(trainedModel, "model.coreml", EModelType::AppleCoreML);
        TFullModel deserializedModel = ReadModel("model.coreml", EModelType::AppleCoreML);
        UNIT_ASSERT_EQUAL(trainedModel.ModelTrees->GetModelTreeData()->GetLeafValues(), deserializedModel.ModelTrees->GetModelTreeData()->GetLeafValues());
        UNIT_ASSERT_EQUAL(trainedModel.ModelTrees->GetModelTreeData()->GetTreeSplits(), deserializedModel.ModelTrees->GetModelTreeData()->GetTreeSplits());
    }

    Y_UNIT_TEST(TestNonSymmetricJsonApply) {
        auto pool = GetAdultPool();
        TDataProviders dataProviders;
        dataProviders.Learn = pool;
        dataProviders.Test.push_back(pool);

        THolder<TLearnProgress> learnProgress;
        NJson::TJsonValue params;
        params.InsertValue("learning_rate", 0.01);
        params.InsertValue("iterations", 100);
        params.InsertValue("random_seed", 1);
        TFullModel trainedModel;
        TEvalResult evalResult;

        TrainModel(
            params,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            dataProviders,
            Nothing(),
            &learnProgress,
            "",
            &trainedModel,
            {&evalResult});

        ExportModel(trainedModel, "oblivious_model.json", EModelType::Json);
        trainedModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        ExportModel(trainedModel, "nonsymmetric_model.json", EModelType::Json);

        TFullModel obliviousModel = ReadModel("oblivious_model.json", EModelType::Json);
        TFullModel nonSymmetricModel = ReadModel("nonsymmetric_model.json", EModelType::Json);
        auto result1 = ApplyModelMulti(obliviousModel, *pool);
        auto result2 = ApplyModelMulti(nonSymmetricModel, *pool);
        UNIT_ASSERT_EQUAL(result1, result2);
    }

    static TString RemoveWhitespacesAndNewLines(const TString& str) {
        TStringBuilder out;
        for (char c : str) {
            if (!EqualToOneOf(c, ' ', '\n')) {
                out << c;
            }
        }
        return out;
    }

    Y_UNIT_TEST(TestNonSymmetricJsonFormat) {
        TFullModel model;
        model.UpdateDynamicData();
        TFloatFeature f0(false, 0, 0, {0.5, 1.5, 2.5});
        TFloatFeature f1(false, 1, 1, {5, 10, 20});
        TFloatFeature f2(false, 2, 2, {5, 15, 25, 35});
        TNonSymmetricTreeModelBuilder builder({f0, f1, f2}, {}, {}, {}, 1);
        {
            auto head = MakeHolder<TNonSymmetricTreeNode>();
            head->SplitCondition = TModelSplit(TFloatSplit(0, 0.5));
            {
                auto left = MakeHolder<TNonSymmetricTreeNode>();
                left->Value = 1.0;
                left->NodeWeight = 10;
                head->Left = std::move(left);
            }
            {
                auto right = MakeHolder<TNonSymmetricTreeNode>();
                right->Value = 2.0;
                right->NodeWeight = 20;
                head->Right = std::move(right);
            }
            builder.AddTree(std::move(head));
        }
        {
            auto head = MakeHolder<TNonSymmetricTreeNode>();
            head->SplitCondition = TModelSplit(TFloatSplit(1, 10));
            {
                auto left = MakeHolder<TNonSymmetricTreeNode>();
                left->SplitCondition = TModelSplit(TFloatSplit(2, 25));
                {
                    auto leftLeft = MakeHolder<TNonSymmetricTreeNode>();
                    leftLeft->Value = 3.0;
                    leftLeft->NodeWeight = 30;
                    left->Left = std::move(leftLeft);
                }
                {
                    auto leftRight = MakeHolder<TNonSymmetricTreeNode>();
                    leftRight->Value = 4.0;
                    leftRight->NodeWeight = 40;
                    left->Right = std::move(leftRight);
                }
                head->Left = std::move(left);
            }
            {
                auto right = MakeHolder<TNonSymmetricTreeNode>();
                right->Value = 5.0;
                right->NodeWeight = 50;
                head->Right = std::move(right);
            }
            builder.AddTree(std::move(head));
        }
        builder.Build(model.ModelTrees.GetMutable());
        const auto json = ConvertModelToJson(model, nullptr, nullptr);
        const TString jsonTreesStr = NJson::WriteJson(&json["trees"], false, true);
        const TString expectedJsonTrees =
            R"([
                {
                "left":
                    {
                    "value":1,
                    "weight":10
                    },
                "right":
                    {
                    "value":2,
                    "weight":20
                    },
                "split":
                    {
                    "border":0.5,
                    "float_feature_index":0,
                    "split_index":0,
                    "split_type":"FloatFeature"
                    }
                },
                {
                "left":
                    {
                    "left":
                        {
                        "value":3,
                        "weight":30
                        },
                    "right":
                        {
                        "value":4,
                        "weight":40
                        },
                    "split":
                        {
                        "border":25,
                        "float_feature_index":2,
                        "split_index":2,
                        "split_type":"FloatFeature"
                        }
                    },
                "right":
                    {
                    "value":5,
                    "weight":50
                    },
                "split":
                    {
                    "border":10,
                    "float_feature_index":1,
                    "split_index":1,
                    "split_type":"FloatFeature"
                    }
                }
            ])";
        UNIT_ASSERT_EQUAL(jsonTreesStr, RemoveWhitespacesAndNewLines(expectedJsonTrees));
    }

    Y_UNIT_TEST(TestNonSymmetricMultiJsonFormat) {
        TFullModel model;
        model.UpdateDynamicData();
        TFloatFeature f0(false, 0, 0, {0.5, 1.5, 2.5});
        TNonSymmetricTreeModelBuilder builder({f0}, {}, {}, {}, 2);
        {
            auto head = MakeHolder<TNonSymmetricTreeNode>();
            head->SplitCondition = TModelSplit(TFloatSplit(0, 0.5));
            {
                auto left = MakeHolder<TNonSymmetricTreeNode>();
                left->Value = TVector<double>{1.0, 2.0};
                left->NodeWeight = 10;
                head->Left = std::move(left);
            }
            {
                auto right = MakeHolder<TNonSymmetricTreeNode>();
                right->Value = TVector<double>{3.0, 4.0};
                right->NodeWeight = 20;
                head->Right = std::move(right);
            }
            builder.AddTree(std::move(head));
        }
        builder.Build(model.ModelTrees.GetMutable());
        const auto json = ConvertModelToJson(model, nullptr, nullptr);
        const TString jsonTreesStr = NJson::WriteJson(&json["trees"], false, true);
        const TString expectedJsonTrees =
            R"([
                {
                "left":
                    {
                    "value":[1, 2],
                    "weight":10
                    },
                "right":
                    {
                    "value":[3,4],
                    "weight":20
                    },
                "split":
                    {
                    "border":0.5,
                    "float_feature_index":0,
                    "split_index":0,
                    "split_type":"FloatFeature"
                    }
                }
            ])";
        assert(jsonTreesStr == RemoveWhitespacesAndNewLines(expectedJsonTrees));
    }
}
