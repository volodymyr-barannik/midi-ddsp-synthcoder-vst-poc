/*
Copyright 2022 The DDSP-VST Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "audio/tflite/ModelLibrary.h"

#include "PredictControlsModel.h"
#include "util/Constants.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace ddsp
{

ModelLibrary::ModelLibrary()
{
    loadEmbeddedModels();
    setPathToUserModels();
    searchPathForModels();
}

ModelLibrary::~ModelLibrary() {}

void ModelLibrary::loadEmbeddedModels()
{
    models.emplace_back (ModelInfo ("Flute",
                                    loadModelTimestamp (BinaryData::Flute_tflite, BinaryData::Flute_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Flute_tflite,
                                    BinaryData::Flute_tfliteSize));

    models.emplace_back (ModelInfo ("Violin",
                                    loadModelTimestamp (BinaryData::Violin_tflite, BinaryData::Violin_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Violin_tflite,
                                    BinaryData::Violin_tfliteSize));

    models.emplace_back (ModelInfo ("Trumpet",
                                    loadModelTimestamp (BinaryData::Trumpet_tflite, BinaryData::Trumpet_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Trumpet_tflite,
                                    BinaryData::Trumpet_tfliteSize));

    models.emplace_back (ModelInfo ("Saxophone",
                                    loadModelTimestamp (BinaryData::Saxophone_tflite, BinaryData::Saxophone_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Saxophone_tflite,
                                    BinaryData::Saxophone_tfliteSize));
    models.emplace_back (ModelInfo ("Bassoon",
                                    loadModelTimestamp (BinaryData::Bassoon_tflite, BinaryData::Bassoon_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Bassoon_tflite,
                                    BinaryData::Bassoon_tfliteSize));
    models.emplace_back (ModelInfo ("Clarinet",
                                    loadModelTimestamp (BinaryData::Clarinet_tflite, BinaryData::Clarinet_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Clarinet_tflite,
                                    BinaryData::Clarinet_tfliteSize));
    models.emplace_back (ModelInfo ("Melodica",
                                    loadModelTimestamp (BinaryData::Melodica_tflite, BinaryData::Melodica_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Melodica_tflite,
                                    BinaryData::Melodica_tfliteSize));
    models.emplace_back (ModelInfo ("Sitar",
                                    loadModelTimestamp (BinaryData::Sitar_tflite, BinaryData::Sitar_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Sitar_tflite,
                                    BinaryData::Sitar_tfliteSize));
    models.emplace_back (ModelInfo ("Trombone",
                                    loadModelTimestamp (BinaryData::Trombone_tflite, BinaryData::Trombone_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Trombone_tflite,
                                    BinaryData::Trombone_tfliteSize));
    models.emplace_back (ModelInfo ("Tuba",
                                    loadModelTimestamp (BinaryData::Tuba_tflite, BinaryData::Tuba_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Tuba_tflite,
                                    BinaryData::Tuba_tfliteSize));
    models.emplace_back (ModelInfo ("Vowels",
                                    loadModelTimestamp (BinaryData::Vowels_tflite, BinaryData::Vowels_tfliteSize),
                                    ModelType::DDSP_v1,
                                    BinaryData::Vowels_tflite,
                                    BinaryData::Vowels_tfliteSize));

    jassert (models.size() == kNumEmbeddedPredictControlsModels);
}

void ModelLibrary::setPathToUserModels()
{
    auto documentsDir = juce::File::getSpecialLocation (juce::File::SpecialLocationType::userDocumentsDirectory);
    pathToUserModels = documentsDir.getChildFile ("Magenta").getChildFile ("DDSP").getChildFile ("Models");
}

int ModelLibrary::getModelIdx (juce::String modelTimestamp)
{
    // If the model exists, return its index, otherwise default to the first one.
    for (int i = 0; i < models.size(); i++)
    {
        if (models[i].timestamp == modelTimestamp)
        {
            return i;
        }
    }
    return 0;
}

juce::String ModelLibrary::getModelTimestamp (int modelIdx) { return models[modelIdx].timestamp; }

juce::File ModelLibrary::getPathToUserModels() { return pathToUserModels; }

std::unique_ptr<tflite::Interpreter> ModelLibrary::getInterpreterForModel(const ModelInfo& modelInfo) const
{
    juce::StringArray errorMsg;

    std::unique_ptr<tflite::FlatBufferModel> modelBuffer;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // Check if the model is able to load.
    modelBuffer = tflite::FlatBufferModel::VerifyAndBuildFromBuffer (modelInfo.data.begin(), modelInfo.data.getSize());

    if (modelBuffer == nullptr)
    {
        errorMsg.add ("Invalid .tflite file.\n");
        showAlertWindow (modelInfo.name, errorMsg);
        return nullptr;
    }

    // Continue setting up model.
    tflite::InterpreterBuilder initBuilder (*modelBuffer, resolver);
    initBuilder (&interpreter);
    jassert (interpreter != nullptr);

    interpreter->SetNumThreads (1);
    jassert (interpreter->AllocateTensors() == kTfLiteOk);

    return interpreter;
}

ModelType getModelType (const ModelInfo& modelInfo, const tflite::Interpreter& modelInterpreter)
{
    const size_t n_inputs = modelInterpreter.inputs().size();
    const size_t n_outputs = modelInterpreter.outputs().size();
    
    if (n_inputs == kNumPredictControlsInputTensors
        && n_outputs == kNumPredictControlsOutputTensors)
    {
        return ModelType::DDSP_v1;
    }

    if (n_inputs == kNumPredictControlsInputTensors_MIDI_DDSP
        && n_outputs == kNumPredictControlsOutputTensors_MIDI_DDSP)
    {
        return ModelType::MIDI_DDSP;
    }
    
    return ModelType::Unknown;
}

// We don't want a call to disk I/O from the plugin on every model load,
// so we will load all models from disk into memory here.
void ModelLibrary::searchPathForModels()
{
    clearUserModels();
    if (pathToUserModels.createDirectory() == juce::Result::ok())
    {
        auto modelArray = pathToUserModels.findChildFiles (juce::File::findFiles, true, "*.tflite");
        models.reserve (modelArray.size());

        for (auto& m : modelArray)
        {
            ModelInfo modelInfo (m.getFileNameWithoutExtension(),
                                 loadModelTimestamp (m.loadFileAsString().toRawUTF8(), m.getSize()),
                                 m.loadFileAsString().toRawUTF8(),
                                 m.getSize());
            
            std::unique_ptr<tflite::Interpreter> interpreter = getInterpreterForModel (modelInfo);

            modelInfo.modelType = getModelType (modelInfo, *interpreter);
            DBG ("Model type: " << static_cast<int>(modelInfo.modelType) << "\n");

            if (validateModel (modelInfo, *interpreter))
                models.emplace_back (modelInfo);
        }
    }
    else
    {
        juce::AlertWindow ("Error",
                           "Could not create directory " + pathToUserModels.getFullPathName(),
                           juce::AlertWindow::AlertIconType::WarningIcon);
    }
}

juce::String ModelLibrary::loadModelTimestamp (const char* modelDataPtr, size_t dataSize)
{
    // Reading model metadata.
    juce::MemoryInputStream modelBufferStream (static_cast<const void*> (modelDataPtr), dataSize, false);
    juce::ZipFile zf (&modelBufferStream, false);

    if (const juce::ZipFile::ZipEntry* e = zf.getEntry ("metadata.json", true))
    {
        if (juce::InputStream* is = zf.createStreamForEntry (*e))
        {
            juce::var json = juce::JSON::parse (is->readEntireStreamAsString());

            DBG (juce::JSON::toString (json));

            auto exportTime = json["export_time"].toString().toUTF8();

            delete is;
            return exportTime;
        }
    }
    else
    {
        DBG ("Cannot access model metadata.");
    }
    // Maybe something else should go here...
    return "";
}

bool ModelLibrary::validateModel (ModelInfo modelInfo, tflite::Interpreter& modelInterpreter) const
{
    juce::StringArray errorMsg;

    int requiredNumOfInputs = -1;
    int requiredNumOfOutputs = -1;
    if (modelInfo.modelType == ModelType::DDSP_v1)
    {
        requiredNumOfInputs = kNumPredictControlsInputTensors;
        requiredNumOfOutputs = kNumPredictControlsOutputTensors;
    }
    else if (modelInfo.modelType == ModelType::MIDI_DDSP)
    {
        requiredNumOfInputs = kNumPredictControlsInputTensors_MIDI_DDSP;
        requiredNumOfOutputs = kNumPredictControlsOutputTensors_MIDI_DDSP;
    }
    else if (modelInfo.modelType == ModelType::Unknown)
    {
        showAlertWindow (modelInfo.name, juce::StringArray(" is of Unknown model type."));
        return false;
    }

    // Check if model has correct number of I/O tensors.
    if (modelInterpreter.inputs().size() != requiredNumOfInputs)
    {
        errorMsg.add ("Invalid number of input tensors: " + std::to_string (modelInterpreter.inputs().size()) + "\n");
    }

    if (modelInterpreter.outputs().size() != requiredNumOfOutputs)
    {
        errorMsg.add ("Invalid number of output tensors: " + std::to_string (modelInterpreter.outputs().size()) + "\n");
    }

    if (! errorMsg.isEmpty())
    {
        showAlertWindow (modelInfo.name, errorMsg);
        return false;
    }

    // Check if tensors have the correct names. Sometimes the colab
    // puts them in different orders so we will stay order-agnostic here.
    // We also do not want repeats.
    juce::StringArray inputNames;
    inputNames.add (PredictControlsModel::getF0InputName(modelInfo).data());
    inputNames.add (PredictControlsModel::getLoudnessInputName(modelInfo).data());
    inputNames.add (PredictControlsModel::getStateInputName(modelInfo).data());

    for (int i = 0; i < modelInterpreter.inputs().size(); i++)
    {
        auto name = modelInterpreter.GetInputName (i);
        
        TfLiteIntArray* shape = modelInterpreter.input_tensor (i)->dims;
        juce::String shape_str = "[";
        for (int d = 0; d < shape->size; ++d)
        {
            shape_str += juce::String (shape->data[d]) + "; ";
        }
        shape_str += "]";

        auto size = modelInterpreter.input_tensor (i)->bytes / sizeof (float);
        DBG ("Model input: [ name = " << name << ", shape = " << shape_str << ", size = " << size << " ]");

        if (int idx = inputNames.indexOf (name); idx != -1)
        {
            inputNames.remove (idx);
        }
        else
        {
            errorMsg.add ("Unknown input tensor name " + std::string (name) + "\n");
        }
    }

    juce::StringArray outputNames;
    outputNames.add (PredictControlsModel::getAmplitudeOutputName(modelInfo).data());
    outputNames.add (PredictControlsModel::getHarmonicsOutputName(modelInfo).data());
    outputNames.add (PredictControlsModel::getNoiseAmpsOutputName(modelInfo).data());
    outputNames.add (PredictControlsModel::getStateOutputName(modelInfo).data());

    for (int i = 0; i < modelInterpreter.outputs().size(); i++)
    {
        auto name = modelInterpreter.GetOutputName (i);

        TfLiteIntArray* shape = modelInterpreter.output_tensor (i)->dims;
        juce::String shape_str = "[";
        for (int d = 0; d < shape->size; ++d)
        {
            shape_str += juce::String (shape->data[d]) + "; ";
        }
        shape_str += "]";

        auto size = modelInterpreter.output_tensor (i)->bytes / sizeof (float);
        DBG ("Model output: [ name = " << name << ", shape = " << shape_str << ", size = " << size << " ]");


        if (int idx = outputNames.indexOf (name); idx != -1)
        {
            outputNames.remove (idx);
        }
        else
        {
            errorMsg.add ("Unknown output tensor name " + std::string (name) + "\n");
        }
    }

    if (!errorMsg.isEmpty())
    {
        showAlertWindow (modelInfo.name, errorMsg);
        //return false;
    }

    // Check if tensors have correct sizes.
    for (int i = 0; i < modelInterpreter.inputs().size(); i++)
    {
        const std::string_view name = modelInterpreter.GetInputName (i);
        auto size = modelInterpreter.input_tensor (i)->bytes / sizeof (float);

        if (name == PredictControlsModel::getF0InputName(modelInfo))
        {
            if (size != kF0Size)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
        else if (name == PredictControlsModel::getLoudnessInputName(modelInfo))
        {
            if (size != kLoudnessSize)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
        else if (name == PredictControlsModel::getStateInputName(modelInfo))
        {
            if (size != kGruModelStateSize)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
    }

    for (int i = 0; i < modelInterpreter.outputs().size(); i++)
    {
        const std::string_view name = modelInterpreter.GetOutputName (i);
        size_t size = modelInterpreter.output_tensor (i)->bytes / sizeof (float);

        if (name == PredictControlsModel::getAmplitudeOutputName(modelInfo))
        {
            if (size != kAmplitudeSize)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
        else if (name == PredictControlsModel::getHarmonicsOutputName(modelInfo))
        {
            if (size != kHarmonicsSize)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
        else if (name == PredictControlsModel::getNoiseAmpsOutputName(modelInfo))
        {
            if (size != kNoiseAmpsSize)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
        else if (name == PredictControlsModel::getStateOutputName(modelInfo))
        {
            if (size != kGruModelStateSize)
            {
                errorMsg.add ("Invalid tensor size " + std::to_string (size) + " for " + name.data() + "\n");
            }
        }
    }

    if (! errorMsg.isEmpty())
    {
        showAlertWindow (modelInfo.name, errorMsg);
        //return false;
    }

    DBG ("Model " << modelInfo.name << " is valid.");

    return true;
}

void ModelLibrary::clearUserModels()
{
    for (size_t i = models.size(); i > kNumEmbeddedPredictControlsModels; --i)
    {
        models.pop_back();
    }
}

void ModelLibrary::showAlertWindow (juce::String modelName, juce::StringArray messages) const
{
    juce::String message;
    for (int i = 0; i < messages.size(); i++)
    {
        message.append (messages[i], messages[i].length());
    }

    juce::NativeMessageBox::showMessageBoxAsync (
        juce::AlertWindow::AlertIconType::WarningIcon, "DDSP - Error loading model: " + modelName, message);
}

} // namespace ddsp
