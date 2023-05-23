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

#include "audio/tflite/PredictControlsModel.h"
#include "util/Constants.h"

#include <random>

namespace ddsp
{

PredictControlsModel::PredictControlsModel (const ModelInfo& mi)
    : ModelBase (mi.data.begin(), mi.data.getSize(), kNumPredictControlsThreads), modelInfo(mi)
{
    reset();
    describe();

    // for random values
    std::random_device rd;
    gen = std::mt19937(rd());
    dis = std::uniform_real_distribution<float>(0.0, 1.0); // Adjust the range of random float values as needed

}

template<typename T>
void initTensorWithRandomValues(TfLiteTensor* tensor, std::uniform_real_distribution<float>& dis, std::mt19937& gen)
{
    int input_size = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        input_size *= tensor->dims->data[i];
    }

    // Cast the input tensor data to float and fill it with random float values
    float* input_data = reinterpret_cast<T*>(tensor->data.raw);
    for (int i = 0; i < input_size; ++i) {
        input_data[i] = dis(gen);
    }
}

void PredictControlsModel::call (const AudioFeatures& input, SynthesisControls& output)
{
    if (modelInfo.modelType == ModelType::MIDI_DDSP)
    {
        DBG ("gathering outputs of MIDI_DDSP model");
    }
    
    for (size_t i = 0; i < interpreter->inputs().size(); ++i)
    {
        const std::string_view inputName (interpreter->GetInputName (i));

        if (inputName == getF0InputName(modelInfo))
        {
            *interpreter->typed_input_tensor<float> (i) = input.f0_norm;
        }
        else if (inputName == getLoudnessInputName(modelInfo))
        {
            *interpreter->typed_input_tensor<float> (i) = input.loudness_norm;
        }
        else if (inputName == getStateInputName(modelInfo))
        {
            for (size_t j = 0; j < gruState.size(); j++)
            {
                interpreter->typed_input_tensor<float> (i)[j] = gruState[j];
            }
        }
        else if (inputName == getMidiInputName (modelInfo))
        {
            initTensorWithRandomValues<float>(interpreter->input_tensor(i), dis, gen);
        }
        else if (inputName == getOnsetsInputName (modelInfo))
        {
            initTensorWithRandomValues<float>(interpreter->input_tensor(i), dis, gen);
        }
        else if (inputName == getOffsetsInputName (modelInfo))
        {
            initTensorWithRandomValues<float>(interpreter->input_tensor(i), dis, gen);
        }
        else if (inputName == getInstrumentIdInputName (modelInfo))
        {
            initTensorWithRandomValues<float>(interpreter->input_tensor(i), dis, gen);
        }
        else
        {
            std::cerr << "Invalid tensor name: " + juce::StringRef (inputName.data()) << std::endl;
        }
    }

    // Run tflite graph computation on input.

    if (const auto status = interpreter->Invoke(); status != kTfLiteOk)
    {
        std::cerr << "Failed to compute, status code: " << status << std::endl;
    }

    size_t n_outputs = interpreter->outputs().size();

    for (size_t output_idx = 0; output_idx < n_outputs; ++output_idx)
    {
        const std::string_view outputName (interpreter->GetOutputName (output_idx));
        
        //if (modelInfo.modelType == ModelType::MIDI_DDSP)
        for (size_t q = 0; q < n_outputs; ++q)
        {
            DBG (juce::String("gathering outputs of MIDI_DDSP model. Output: ") + juce::String(std::string(outputName)));
            TfLiteTensor* output_tensor0 = interpreter->output_tensor (q);
            float* output_tensor0_2 = interpreter->typed_output_tensor<float> (q);
            int* output_tensor0_3 = interpreter->typed_output_tensor<int> (q);
            double* output_tensor0_4 = interpreter->typed_output_tensor<double> (q);
            
            char* data_raw = output_tensor0->data.raw;
            float* output_tensor0_rc = reinterpret_cast<float*> (data_raw);

            DBG (int (data_raw));
            DBG (int (output_tensor0));
            DBG ((output_tensor0_2 ? *output_tensor0_2 : 0.f));
            DBG ((output_tensor0_3 ? *output_tensor0_3 : 0));
            DBG ((output_tensor0_4 ? *output_tensor0_4 : 0.0));
            DBG ((output_tensor0_rc ? *output_tensor0_rc : 0.0));
        }

        if (outputName == getAmplitudeOutputName (modelInfo))
        {
            output.amplitude = *interpreter->typed_output_tensor<float> (output_idx);
        }
        else if (outputName == getHarmonicsOutputName (modelInfo))
        {
            for (int j = 0; j < kHarmonicsSize; j++)
            {
                output.harmonics[j] = interpreter->typed_output_tensor<float> (output_idx)[j];
            }
        }
        else if (outputName == getNoiseAmpsOutputName (modelInfo))
        {
            for (size_t j = 0; j < kNoiseAmpsSize; j++)
            {
                output.noiseAmps[j] = interpreter->typed_output_tensor<float> (output_idx)[j];
            }
        }
        else if (outputName == getStateOutputName (modelInfo))
        {
            for (size_t j = 0; j < gruState.size(); j++)
            {
                gruState[j] = interpreter->typed_output_tensor<float> (output_idx)[j];
            }
        }
        else
        {
            std::cerr << "Invalid tensor name: " + juce::StringRef (outputName.data()) << std::endl;
        }
    }

    for (size_t i = 0; i < kHarmonicsSize; ++i)
    {
        if (isnan (output.harmonics[i]))
        {
            DBG ("is_nan");
            output.harmonics[i] = 0.f;
            output.amplitude = 0.f;
        }
    }

    output.f0_hz = input.f0_hz;
}

void PredictControlsModel::reset()
{
    juce::FloatVectorOperations::clear (gruState.data(), static_cast<int> (gruState.size()));
}

const PredictControlsModel::Metadata PredictControlsModel::getMetadata (const ModelInfo& mi)
{
    PredictControlsModel::Metadata metadata;
    // read model metadata
    juce::MemoryInputStream modelBufferStream (static_cast<const void*> (mi.data.begin()), mi.data.getSize(), false);
    juce::ZipFile zf (&modelBufferStream, false);

    if (const juce::ZipFile::ZipEntry* e = zf.getEntry ("metadata.json", true))
    {
        if (juce::InputStream* is = zf.createStreamForEntry (*e))
        {
            juce::var json = juce::JSON::parse (is->readEntireStreamAsString());

            DBG (juce::JSON::toString (json));

            metadata.minPitch_Hz = json["mean_min_pitch_note_hz"];
            metadata.maxPitch_Hz = json["mean_max_pitch_note_hz"];
            metadata.minPower_dB = json["mean_min_power_note"];
            metadata.maxPower_dB = json["mean_max_power_note"];
            metadata.version = json["version"].toString().toUTF8();
            metadata.exportTime = json["export_time"].toString().toUTF8();

            delete is;
        }
    }
    else
    {
        DBG ("Cannot access model metadata.");
    }

    return metadata;
}

std::string_view PredictControlsModel::getF0InputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kInputTensorName_F0
                                           : kInputTensorName_F0_MIDI_DDSP;
}

std::string_view PredictControlsModel::getLoudnessInputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kInputTensorName_Loudness
                                           : kInputTensorName_Loudness_MIDI_DDSP;
}

std::optional<std::string_view> PredictControlsModel::getMidiInputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::MIDI_DDSP ? std::optional(kInputTensorName_Midi_MIDI_DDSP) 
                                                       : std::nullopt;
}

std::optional<std::string_view> PredictControlsModel::getOnsetsInputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::MIDI_DDSP ? std::optional(kInputTensorName_Onsets_MIDI_DDSP) 
                                                       : std::nullopt;
}

std::optional<std::string_view> PredictControlsModel::getOffsetsInputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::MIDI_DDSP ? std::optional (kInputTensorName_Offsets_MIDI_DDSP)
                                                       : std::nullopt;
}

std::optional<std::string_view> PredictControlsModel::getInstrumentIdInputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::MIDI_DDSP ? std::optional (kInputTensorName_InstrumentId_MIDI_DDSP)
                                                       : std::nullopt;
}

std::string_view PredictControlsModel::getStateInputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kInputTensorName_State
                                           : kInputTensorName_State_MIDI_DDSP;
}

std::string_view PredictControlsModel::getAmplitudeOutputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kOutputTensorName_Amplitude
                                           : kOutputTensorName_Amplitude_MIDI_DDSP;
}

std::string_view PredictControlsModel::getHarmonicsOutputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kOutputTensorName_Harmonics
                                           : kOutputTensorName_Harmonics_MIDI_DDSP;
}

std::string_view PredictControlsModel::getNoiseAmpsOutputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kOutputTensorName_NoiseAmps
                                           : kOutputTensorName_NoiseAmps_MIDI_DDSP;
}

std::string_view PredictControlsModel::getStateOutputName (const ModelInfo& modelInfo)
{
    return modelInfo.modelType == ModelType::DDSP_v1 ? kOutputTensorName_State
                                           : kOutputTensorName_State_MIDI_DDSP;
}
} // namespace ddsp
