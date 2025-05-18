from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Cropping1D
import pandas as pd
import numpy as np
import pybaselines
from scipy.stats import pearsonr
from scipy import interpolate
import logging
from io import StringIO
import os
import tensorflow as tf # Ensure tf is imported if used directly
from pydantic import BaseModel # Ensure BaseModel is imported


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define constants ---
# Target length for all processing pipeline steps (preprocessing, classification, final output)
# This matches the length of SynCleanSet.npy as per user requirement.
TARGET_SPEC_LENGTH = 6501
# Length specifically required by the models (observed from previous errors for SOME models)
# We will check the model's actual required length dynamically in denoise.
# Keeping this constant here as a default or reference is okay, but the denoise endpoint logic is key.
MODEL_INPUT_LENGTH_DEFAULT = 6496 # Renamed to default as it might not apply to all models


# Wavenumber array for the target processing length (6501 points)
# Using np.arange(700, 3950.5, 0.5) which gives 6501 points exactly over this range.
WaveRef = np.arange(700, 3950.5, 0.5)
# Defensive check just in case arange behaves unexpectedly
if len(WaveRef) != TARGET_SPEC_LENGTH:
     logger.error(f"Generated WaveRef length {len(WaveRef)} != TARGET_SPEC_LENGTH {TARGET_SPEC_LENGTH}. Using linspace as fallback.")
     WaveRef = np.linspace(700, 3950.5, TARGET_SPEC_LENGTH)

# We will generate model-specific WaveRefs inside denoise if needed,
# based on the dynamically determined model input length.


# --- Define materials and sample counts ---
NameList = ["Acrylic", "Cellulose", "ENR", "EPDM", "HDPE", "LDPE", "Nylon", "PBAT", "PBS", "PC",
            "PEEK", "PEI", "PET", "PLA", "PMMA", "POM", "PP", "PS", "PTFE", "PU", "PVA", "PVC"]
NumSample = 200 # Unused in the provided code snippets
NumCleanSpec = 10 # Number of clean spectra per material *expected* in SynCleanSet for grouping


# --- Load SynCleanSet dataset ---
# This file is expected to contain clean reference spectra with TARGET_SPEC_LENGTH points (6501).
try:
    SynCleanSet = np.load("SynCleanSet.npy")
    logger.info(f"Successfully loaded SynCleanSet.npy with shape {SynCleanSet.shape}")

    # Perform checks on the loaded SynCleanSet
    if len(SynCleanSet.shape) != 2:
        logger.error(f"SynCleanSet.npy is not a 2D array. Expected shape (TotalSamples, SpectrumLength), got {SynCleanSet.shape}.")
        raise ValueError("Reference dataset (SynCleanSet.npy) has incorrect dimensions. Expected a 2D array.")

    synset_spectrum_length = SynCleanSet.shape[1]
    total_synset_samples = SynCleanSet.shape[0]
    expected_total_samples = len(NameList) * NumCleanSpec

    # CRITICAL CHECK: The length of spectra in SynCleanSet must match the TARGET_SPEC_LENGTH.
    # This check should now pass if SynCleanSet.npy has 6501 points.
    if synset_spectrum_length != TARGET_SPEC_LENGTH:
        logger.error(f"Mismatch between pipeline target length ({TARGET_SPEC_LENGTH}) and SynCleanSet spectrum length ({synset_spectrum_length}).")
        raise HTTPException(status_code=500,
                            detail=f"Configuration Error: Reference dataset (SynCleanSet.npy) has spectra of length {synset_spectrum_length}, but the pipeline target length is {TARGET_SPEC_LENGTH}. These must match.")

    # Warning if the total number of samples doesn't match the expected grouping
    if total_synset_samples < expected_total_samples:
         logger.warning(f"SynCleanSet.npy has fewer rows ({total_synset_samples}) than expected ({expected_total_samples} = {len(NameList)} materials * {NumCleanSpec} samples/material). Classification might not cover all materials or samples as intended.")
    elif total_synset_samples > expected_total_samples:
         logger.warning(f"SynCleanSet.npy has more rows ({total_synset_samples}) than expected ({expected_total_samples}). Only the first {expected_total_samples} rows will be used for standard material grouping in classification.")


except Exception as e:
    logger.error(f"Error loading SynCleanSet.npy: {str(e)}")
    if not isinstance(e, HTTPException):
         raise HTTPException(status_code=500, detail=f"Error loading reference dataset: {str(e)}")
    else:
         raise e


# --- Load models ---
# Ensure custom objects include LeakyReLU, as it's used in the models
custom_objects = {
    "LeakyReLU": LeakyReLU(alpha=0.2),
    # Add other necessary custom objects if any standard Keras layers are custom
    # "Dense": Dense, "Conv1D": Conv1D, etc. are usually not needed here unless subclassed
}

model_dir = os.path.join(os.path.dirname(__file__), "model_files")

model_paths = {
    "Autoencoder": os.path.join(model_dir, "model_AE-10db.h5"),
    "Unet": os.path.join(model_dir, "model_Unet-10db.h5"),
    "CNN-Autoencoder (CNNAE)": os.path.join(model_dir, "model_CNNAE-10db.h5"),
    "CNNAE-MobileNet": os.path.join(model_dir, "model_CNNAE-MobileNet-10db.h5"),
    "CNNAE-ResNet50": os.path.join(model_dir, "model_CNNAE-ResNet50-10db.h5"),
    "CNNAE-InceptionV3": os.path.join(model_dir, "model_CNNAE-InceptionV3-10db.h5"),
    "CNNAE-InceptionResNet": os.path.join(model_dir, "model_CNNAE-InceptionResNet-10db.h5"),
    "CNNAE-Xception": os.path.join(model_dir, "model_CNNAE-Xception-10db.h5"),
}

models_loaded = {}

for model_name, path in model_paths.items():
    try:
        # compile=False is correct for loading a pre-trained model for inference
        model = load_model(path, custom_objects=custom_objects, compile=False)
        models_loaded[model_name] = model # Store the loaded model

        logger.info(f"Successfully loaded model: {model_name}")

        # --- Corrected Check model input shape to understand what the model expects ---
        model_input_shape = None
        if hasattr(model, 'input_shape') and model.input_shape is not None:
            model_input_shape = model.input_shape
        elif hasattr(model, 'inputs') and model.inputs and model.inputs[0].shape is not None:
            # For models loaded via functional API, input shape might be in inputs[0].shape
            # Convert TensorShape to tuple, handling None dimensions
            model_input_shape = tuple(dim if isinstance(dim, int) else None for dim in model.inputs[0].shape)

        if model_input_shape is not None:
            logger.info(f"Model '{model_name}' input shape: {model_input_shape}")

            # Try to determine the required sequence length and dimension count from the shape tuple
            required_len = None
            required_dims = len(model_input_shape)

            # The sequence length is usually the second element (index 1) after the batch dimension (index 0)
            if required_dims >= 2:
                 required_len = model_input_shape[1]

            if required_len is None or required_dims < 2: # Check if length was found and dimensions make sense
                logger.warning(f"Could not reliably determine input sequence length or dimensions (expected >=2) for model '{model_name}' from shape {model_input_shape}. This model might fail prediction.")
                # Store None or some indicator if shape is ambiguous
                models_loaded[model_name + "_input_info"] = {'length': None, 'dims': None}
            else:
                 # Store the required info
                 models_loaded[model_name + "_input_info"] = {'length': required_len, 'dims': required_dims}
                 if required_len != TARGET_SPEC_LENGTH:
                     logger.warning(f"Model '{model_name}' input length ({required_len}) differs from pipeline target length ({TARGET_SPEC_LENGTH}). Interpolation will be needed before feeding input to this model.")
                 if required_dims == 2:
                      # Check if it's a 2D shape like (None, length) - common for simple models but Conv1D needs 3D
                     logger.warning(f"Model '{model_name}' expects 2D input shape {model_input_shape}. Input will be reshaped to 2D (batch, length) before prediction.")
                 elif required_dims == 3 and model_input_shape[2] != 1:
                     # Check if it's a 3D shape with more than 1 channel - unexpected for simple spectra
                     logger.warning(f"Model '{model_name}' expects 3D input shape {model_input_shape} with {model_input_shape[2]} channels. Input will be reshaped to (1, length, 1) assuming 1 channel.")


        else:
             logger.warning(f"Could not determine input shape for model '{model_name}'. This model might fail prediction.")
             models_loaded[model_name + "_input_info"] = {'length': None, 'dims': None}


    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}", exc_info=True)
        # Re-raise the exception as an HTTP exception for the API client
        # This happens during startup if a model file is bad or incompatible
        raise HTTPException(status_code=500, detail=f"Error loading model {model_name}: {str(e)}")


# --- Helper Functions ---

class SpectrumData(BaseModel):
    wavenumbers: list[float]
    intensities: list[float]

def BaseLineCorrection(Spec):
    try:
        if len(Spec) <= 6: # poly_order=6 requires at least 7 points
             logger.warning(f"Spectrum too short ({len(Spec)} pts) for baseline correction (poly_order=6). Skipping.")
             return Spec
        # Ensure Spec is numpy array before calling pybaselines
        Spec_np = np.array(Spec, dtype=np.float32)
        # Handle potential non-finite values before baseline correction
        if not np.isfinite(Spec_np).all():
             logger.warning("Spectrum contains non-finite values before baseline correction. Replacing with 0.")
             Spec_np = np.nan_to_num(Spec_np, nan=0.0, posinf=0.0, neginf=0.0)

        BLine = pybaselines.polynomial.imodpoly(Spec_np, poly_order=6)[0]
        Spec_BC = Spec_np - BLine
        return Spec_BC
    except Exception as e:
        logger.error(f"Baseline correction failed: {str(e)}", exc_info=True)
        # Raise a RuntimeError here, the endpoint calling this will catch it and convert to HTTPException
        raise RuntimeError("Baseline correction failed") from e

def SpecInterpolation(WaveOrigin, Spec, WaveOut):
    """
    Interpolates a spectrum from one wavenumber range/resolution to another.
    Args:
        WaveOrigin (np.ndarray): Original wavenumbers.
        Spec (np.ndarray): Original intensities corresponding to WaveOrigin.
        WaveOut (np.ndarray): Target wavenumbers for interpolation.
    Returns:
        np.ndarray: Interpolated spectrum intensities corresponding to WaveOut.
    Raises:
        RuntimeError: If interpolation fails due to invalid input or unexpected errors.
    """
    try:
        # Ensure inputs are numpy arrays with appropriate dtype
        WaveOrigin_np = np.array(WaveOrigin, dtype=np.float32)
        Spec_np = np.array(Spec, dtype=np.float32)
        WaveOut_np = np.array(WaveOut, dtype=np.float32)

        # Ensure WaveOrigin is sorted for interpolation
        sorted_indices = np.argsort(WaveOrigin_np)
        WaveOrigin_sorted = WaveOrigin_np[sorted_indices]
        Spec_sorted = Spec_np[sorted_indices]

        # Check for basic validity of input wavenumber range
        if len(np.unique(WaveOrigin_sorted)) < 2:
             raise ValueError(f"Original wavenumbers have fewer than 2 unique points ({len(np.unique(WaveOrigin_sorted))}). Cannot interpolate.")
        if WaveOrigin_sorted[0] >= WaveOrigin_sorted[-1]:
             logger.warning("Original wavenumbers are not strictly increasing after sorting.")
             # Interpolation might still work, but it's unusual data

        # Check if any wave point or intensity is NaN or Inf
        if not np.isfinite(WaveOrigin_sorted).all():
             raise ValueError("Original wavenumbers contain non-finite values (NaN or Inf).")
        # It's safer to handle non-finite intensities by replacing them before interpolation
        if not np.isfinite(Spec_sorted).all():
             logger.warning("Original intensities contain non-finite values (NaN or Inf). Replacing with 0 for interpolation.")
             Spec_sorted = np.nan_to_num(Spec_sorted, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.isfinite(WaveOut_np).all():
              raise ValueError("Target wavenumbers contain non-finite values (NaN or Inf).")

        # Use 'linear' interpolation, extrapolate values outside the original range
        Spec_Inter = interpolate.interp1d(WaveOrigin_sorted, Spec_sorted, kind='linear', fill_value="extrapolate")
        Spec_New = Spec_Inter(WaveOut_np)

        # Replace any NaNs resulting from extrapolation with zeros (shouldn't happen with 'extrapolate' and finite inputs but good practice)
        if np.isnan(Spec_New).any():
             logger.warning("Interpolation produced NaN values in output. Replacing with zeros.")
             Spec_New = np.nan_to_num(Spec_New, nan=0.0)
        # Also check for Infs in output
        if not np.isfinite(Spec_New).all():
             logger.warning("Interpolation produced Inf values in output. Replacing with zeros.")
             Spec_New = np.nan_to_num(Spec_New, posinf=0.0, neginf=0.0)


        return Spec_New
    except Exception as e:
        logger.error(f"Interpolation failed: {str(e)}", exc_info=True)
        # Catch specific value errors for better detail
        if isinstance(e, ValueError):
             raise RuntimeError(f"Spectrum interpolation failed: {e}") from e
        raise RuntimeError("Spectrum interpolation failed unexpectedly") from e

def MinMaxNorm(Spec):
    # Ensure Spec is numpy array with appropriate dtype
    Spec_np = np.array(Spec, dtype=np.float32)
    S_Max = np.max(Spec_np)
    S_Min = np.min(Spec_np)
    if S_Max == S_Min:
        # Return zeros if spectrum is flat to avoid division by zero
        return np.zeros_like(Spec_np)
    Spec_Norm = (Spec_np - S_Min) / (S_Max - S_Min)
    # Ensure output is finite after normalization (e.g., if max/min were Inf or NaN)
    if not np.isfinite(Spec_Norm).all():
         logger.warning("Normalization produced non-finite values (NaN or Inf). Replacing with zeros.")
         Spec_Norm = np.nan_to_num(Spec_Norm, nan=0.0, posinf=0.0, neginf=0.0)
    return Spec_Norm

def input_spectrum(file: UploadFile):
    """
    Reads a CSV file from UploadFile, parses it into wavenumbers and intensities, sorts by wavenumber.
    """
    try:
        # Read file contents
        contents = file.file.read()

        # Attempt to decode contents, trying common encodings
        try:
            csv_string = contents.decode('utf-8')
        except UnicodeDecodeError:
             logger.warning("Failed to decode file as UTF-8. Trying 'latin-1'.")
             csv_string = contents.decode('latin-1')
        except Exception as e:
             raise ValueError(f"Could not decode file content: {e}") from e

        # Use StringIO to treat the string as a file for pandas
        csv_data = StringIO(csv_string)

        # Read CSV, assuming no header and two columns
        # Set dtype to float to prevent inference issues with mixed types and automatically handle comma decimals if locale is set
        df = pd.read_csv(csv_data, header=None, dtype=float)

        # Validate number of columns
        if len(df.columns) != 2:
            raise ValueError("Invalid CSV format: Expected exactly two columns (Wavenumber, Intensity).")

        df.columns = ['Wavenumber', 'Intensity']

        # Check for non-finite values (NaN, Inf) which can result from dtype=float if source data is messy
        if not np.isfinite(df['Wavenumber']).all() or not np.isfinite(df['Intensity']).all():
            # Attempt to convert strictly to see if it throws an error
            try:
                 pd.to_numeric(df['Wavenumber'], errors='raise')
                 pd.to_numeric(df['Intensity'], errors='raise')
            except ValueError as e:
                 raise ValueError(f"Invalid data format in CSV: Non-numeric or problematic values found in Wavenumber or Intensity columns: {e}")
            except Exception: # Catch other potential issues during strict conversion
                 pass # If strict conversion didn't fail, the non-finite must have come from pandas float coercion - handle below

            # If still here, non-finite values exist that weren't caught by strict conversion, or strict conversion isn't desired
            logger.warning("CSV data contains non-finite values (NaN/Inf). These will be handled during preprocessing/interpolation.")
            # Decide if you want to clean them here or let interpolation/normalization handle them later.
            # Let's let interpolation/normalization handle them as they have specific handling.


        # Sort by Wavenumber as interpolation requires sorted data
        df = df.sort_values('Wavenumber').reset_index(drop=True)

        wavenumber = df['Wavenumber'].tolist()
        intensity = df['Intensity'].tolist()

        # Check if spectum is empty or too short
        if not wavenumber or len(wavenumber) < 2:
             raise ValueError("Input spectrum is empty or contains fewer than 2 points.")


        return {
            "wavenumber": wavenumber,
            "intensity": intensity
        }
    except Exception as e:
        # Catch specific pandas/IO errors and wrap them in a ValueError
        if isinstance(e, (pd.errors.EmptyDataError, pd.errors.ParserError)):
             raise ValueError(f"Error reading or parsing CSV file: {e}") from e
        # Catch ValueErrors raised internally (e.g., invalid format, decoding, empty spectrum)
        if isinstance(e, ValueError):
             raise e
        # Catch any other unexpected errors
        raise ValueError(f"An unexpected error occurred while processing the CSV file: {str(e)}") from e


# --- API Endpoints ---

@app.post("/api/input")
async def get_input_spectrum(file: UploadFile = File(...)):
    """
    Uploads a CSV file and returns the raw spectrum data (wavenumbers and intensities).
    """
    try:
        data = input_spectrum(file)
        return JSONResponse(content={
            'inputSpectrum': {
                'wavenumbers': data['wavenumber'],
                'intensities': data['intensity']
            }
        })
    except ValueError as e:
        logger.error(f"Input spectrum processing error: {e}")
        # Use status code 400 for client-side errors (bad input)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
         logger.error(f"Unexpected error in /api/input: {e}", exc_info=True)
         # Use status code 500 for unexpected server errors
         raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the input file.")


@app.post("/api/preprocess")
async def get_preprocessed_spectrum(
    file: UploadFile = File(...),
    preprocessing_option: str = Form("baselineCorrection")
):
    """
    Uploads a CSV file, interpolates it to the target pipeline length (TARGET_SPEC_LENGTH = 6501),
    and applies selected preprocessing (baseline correction, normalization, or both).
    Returns the preprocessed spectrum (6501 points).
    """
    try:
        data = input_spectrum(file)
        WaveOrigin = np.array(data['wavenumber'])
        Spec = np.array(data['intensity'])

        # Always interpolate to the target pipeline wavelength range/length (TARGET_SPEC_LENGTH = 6501 points)
        # Use WaveRef as the target wavenumbers
        Spec_interpolated = SpecInterpolation(WaveOrigin, Spec, WaveRef)
        WaveProcessed = WaveRef # Wavenumbers for the preprocessed spectrum are now the target ones

        logger.info(f"Interpolated spectrum to length {len(Spec_interpolated)} ({TARGET_SPEC_LENGTH}).")

        # Apply selected preprocessing steps to the interpolated spectrum
        Spec_preprocessed = Spec_interpolated # Start with the interpolated spectrum
        if preprocessing_option == "baselineCorrection":
            Spec_preprocessed = BaseLineCorrection(Spec_preprocessed)
            logger.info("Applied baseline correction.")
        elif preprocessing_option == "minMaxNormalization":
            Spec_preprocessed = MinMaxNorm(Spec_preprocessed)
            logger.info("Applied Min-Max normalization.")
        elif preprocessing_option == "baselineMinMax":
            Spec_preprocessed = BaseLineCorrection(Spec_preprocessed)
            Spec_preprocessed = MinMaxNorm(Spec_preprocessed)
            logger.info("Applied baseline correction and Min-Max normalization.")
        elif preprocessing_option == "noPreprocessing":
             # Interpolation was still applied for consistency of length
             logger.info("No preprocessing steps applied (interpolation performed to standard length).")
        else:
            raise ValueError(f"Invalid preprocessing option: {preprocessing_option}. Must be one of: baselineCorrection, minMaxNormalization, baselineMinMax, noPreprocessing.")

        # Ensure the final preprocessed spectrum is the correct target length
        if len(Spec_preprocessed) != TARGET_SPEC_LENGTH:
             # This indicates a problem in the interpolation or BaseLineCorrection modifying length (unlikely for BaseLine)
             logger.error(f"Preprocessing output length {len(Spec_preprocessed)} != TARGET_SPEC_LENGTH {TARGET_SPEC_LENGTH}. This is unexpected.")
             # Attempt to fix length with interpolation, but log the error
             try:
                 logger.warning("Attempting to re-interpolate preprocessed output to correct length.")
                 # Create temp wave ref for the current incorrect length, then interpolate onto the correct WaveRef
                 temp_wrong_waveref = np.linspace(WaveRef[0], WaveRef[-1], len(Spec_preprocessed))
                 Spec_preprocessed = SpecInterpolation(temp_wrong_waveref, Spec_preprocessed, WaveRef)
             except Exception as re_interp_e:
                  logger.error(f"Failed to re-interpolate preprocessed output: {re_interp_e}")
                  raise RuntimeError("Preprocessing output length incorrect and failed to fix.") from re_interp_e
             # After attempting fix, check again
             if len(Spec_preprocessed) != TARGET_SPEC_LENGTH:
                  logger.error(f"Re-interpolation failed, output length still incorrect ({len(Spec_preprocessed)}).")
                  raise RuntimeError("Preprocessing output length incorrect and could not be fixed.")


        return JSONResponse(content={
            'preprocessedSpectrum': {
                'wavenumbers': WaveProcessed.tolist(),
                'intensities': Spec_preprocessed.tolist()
            }
        })
    except (ValueError, RuntimeError) as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        status_code = 400 if isinstance(e, ValueError) else 500 # Use 400 for input errors, 500 for runtime/interpolation
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /api/preprocess: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during preprocessing.")


@app.post("/api/denoise")
async def denoise_spectrum_endpoint(
    intensities: str = Form(...), # Expecting a JSON string of the intensity list (preprocessed, TARGET_SPEC_LENGTH = 6501)
    denoising_option: str = Form(...),
    model_name: str = Form(...)
):
    """
    Denoises a spectrum using a selected model.
    Expects preprocessed intensities (already interpolated to TARGET_SPEC_LENGTH = 6501).
    Interpolates to the model's required input length, runs the model, and interpolates output back to TARGET_SPEC_LENGTH.
    """
    try:
        logger.info(f"Denoising request - model: {model_name}, option: {denoising_option}")

        intensities_list = json.loads(intensities)
        # Input spectrum from preprocessing should be TARGET_SPEC_LENGTH (6501)
        Spec_input_target_length = np.array(intensities_list, dtype=np.float32)

        # Validate input length for the denoising step
        if len(Spec_input_target_length) != TARGET_SPEC_LENGTH:
             logger.error(f"Input spectrum length {len(Spec_input_target_length)} != target length {TARGET_SPEC_LENGTH} for denoising.")
             raise HTTPException(status_code=400, detail=f"Input spectrum length {len(Spec_input_target_length)} is incorrect. Expected {TARGET_SPEC_LENGTH} points.")

        # If denoising is disabled, just return the input (which is already TARGET_SPEC_LENGTH)
        if denoising_option == "Disable":
            logger.info("Denoising disabled. Returning original intensities (length %d).", len(Spec_input_target_length))
            return JSONResponse(content={'denoisedSpectrum': Spec_input_target_length.tolist()})

        model = models_loaded.get(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or failed to load.")

        # --- Dynamically determine model input requirements ---
        model_input_info = models_loaded.get(model_name + "_input_info")
        if model_input_info is None or model_input_info['length'] is None:
             logger.error(f"Input shape information not available for model '{model_name}'. Cannot prepare input.")
             raise HTTPException(status_code=500, detail=f"Internal Error: Model '{model_name}' input shape information is missing or ambiguous.")

        required_model_input_length = model_input_info['length']
        required_model_input_dims = model_input_info['dims']

        # --- Prepare spectrum for the model's SPECIFIC input length and shape ---
        spec_to_model_process = Spec_input_target_length # Start with the 6501-point input

        # Interpolate if the model requires a different input length
        if len(spec_to_model_process) != required_model_input_length:
            logger.debug(f"Interpolating spectrum from {len(spec_to_model_process)} to model input length {required_model_input_length} for prediction.")
            # Create a temporary wavenumber array for the required model input length, covering the same range as WaveRef
            temp_model_wave_ref = np.linspace(WaveRef[0], WaveRef[-1], required_model_input_length)
            spec_to_model_process = SpecInterpolation(WaveRef, spec_to_model_process, temp_model_wave_ref)
        else:
             logger.debug(f"Input spectrum length {len(spec_to_model_process)} matches model input length {required_model_input_length}. No interpolation needed before model.")


        # Reshape for model input based on required dimensions
        # Keras models expect a batch dimension (usually None or 1) as the first dimension.
        # We need to add a batch dimension of 1.
        # If the model expects 3D (batch, length, channels), add a channel dimension of 1.
        if required_model_input_dims == 3:
             spec_for_model_input = spec_to_model_process.reshape(1, required_model_input_length, 1)
             logger.debug(f"Reshaped input to 3D {spec_for_model_input.shape} for model.")
        elif required_model_input_dims == 2:
             spec_for_model_input = spec_to_model_process.reshape(1, required_model_input_length)
             logger.debug(f"Reshaped input to 2D {spec_for_model_input.shape} for model.")
        else:
             # This case should be caught by the initial model_input_info check, but as a safeguard
             logger.error(f"Model '{model_name}' requires {required_model_input_dims} dimensions, which is not supported for input preparation.")
             raise HTTPException(status_code=500, detail=f"Internal Error: Unsupported input dimension requirement ({required_model_input_dims}D) for model '{model_name}'.")


        # --- Denoise using the model ---
        denoised_output_raw = model.predict(spec_for_model_input)
        logger.debug(f"Raw model output shape: {denoised_output_raw.shape}")


        # --- Post-process the denoised output (from model length) ---
        # Model output needs to be converted back to a 1D array.
        # The length of the output might be the same as the input length (required_model_input_length),
        # or slightly different depending on the model architecture (e.g., padding in convolutions).

        denoised_model_output_1d = np.array(denoised_output_raw, dtype=np.float32).reshape(-1)
        model_output_length = len(denoised_model_output_1d)

        # --- Interpolate the denoised output BACK UP to the TARGET_SPEC_LENGTH ---
        denoised_target_length = None
        if model_output_length != TARGET_SPEC_LENGTH:
             logger.debug(f"Interpolating denoised output from {model_output_length} back to target length {TARGET_SPEC_LENGTH}.")
             # Create a temporary wavenumber array for the actual model output length, covering the same range as WaveRef
             # Assuming the model output range corresponds to the same range as the input, just potentially different points
             temp_model_output_wave_ref = np.linspace(WaveRef[0], WaveRef[-1], model_output_length)
             denoised_target_length = SpecInterpolation(temp_model_output_wave_ref, denoised_model_output_1d, WaveRef)
        else:
             logger.debug(f"Denoised output length {model_output_length} matches target length {TARGET_SPEC_LENGTH}. No interpolation needed after model.")
             denoised_target_length = denoised_model_output_1d


        # Ensure the final output is TARGET_SPEC_LENGTH (should be if interpolation worked)
        if len(denoised_target_length) != TARGET_SPEC_LENGTH:
             logger.error(f"Final denoised spectrum length {len(denoised_target_length)} != TARGET_SPEC_LENGTH {TARGET_SPEC_LENGTH} after processing.")
             # This indicates a critical failure in interpolation or unexpected data
             raise HTTPException(status_code=500, detail=f"Internal Error: Final denoised spectrum length mismatch ({len(denoised_target_length)} vs {TARGET_SPEC_LENGTH}).")


        # Apply Min-Max normalization to the final denoised spectrum (6501 points)
        denoised_target_length = MinMaxNorm(denoised_target_length)
        logger.info(f"Denoised spectrum processed and normalized. Final Length: {len(denoised_target_length)}")

        # Return the denoised spectrum as a list of intensities (TARGET_SPEC_LENGTH = 6501 points)
        return JSONResponse(content={'denoisedSpectrum': denoised_target_length.tolist()})

    except (ValueError, json.JSONDecodeError, RuntimeError) as e:
        logger.error(f"Denoising error: {e}", exc_info=True)
        status_code = 400 if isinstance(e, (ValueError, json.JSONDecodeError)) else 500 # Use 400 for input/parsing, 500 for runtime/interpolation
        raise HTTPException(status_code=status_code, detail=str(e))
    except HTTPException as e:
         # Re-raise HTTP exceptions raised intentionally
         raise e
    except Exception as e:
        logger.error(f"Denoising error: {str(e)}", exc_info=True)
        # Check if the error is due to shape incompatibility during prediction explicitly
        # The specific error message might vary slightly, but this captures common ones
        if "incompatible with the layer" in str(e) or "Input 0 of layer" in str(e) or "could not be broadcast" in str(e) or "None is not a valid shape" in str(e):
             # Check for common Keras/TF shape errors
             detail = f"Model prediction error due to shape mismatch. Ensure input processing aligns with model requirements. Original error: {e}"
             # Log model input shape info if available to help diagnose
             model_input_info = models_loaded.get(model_name + "_input_info")
             if model_input_info:
                 detail += f" Model '{model_name}' reported expecting {model_input_info.get('dims')}D input of length {model_input_info.get('length')}."
             raise HTTPException(status_code=500, detail=detail) from e
        else:
             raise HTTPException(status_code=500, detail=f"An unexpected error occurred during denoising: {str(e)}") from e


@app.post("/api/classify")
async def classify_spectrum_endpoint(
    intensities: str = Form(...), # Expecting a JSON string of the intensity list (denoised, TARGET_SPEC_LENGTH = 6501)
    model_name: str = Form(...) # model_name is just for reporting which model was used for denoising
):
    """
    Classifies a spectrum by comparing it to average clean spectra using Pearson correlation.
    Expects intensities of a preprocessed/denoised spectrum (already interpolated to TARGET_SPEC_LENGTH = 6501).
    Compares against SynCleanSet which is also expected to have TARGET_SPEC_LENGTH = 6501.
    """
    try:
        logger.info(f"Classification request (using denoised spectrum from model: {model_name})")

        # Parse the intensities JSON string
        intensities_list = json.loads(intensities)
        # Convert to numpy array with float32 dtype
        denoised_spectrum = np.array(intensities_list, dtype=np.float32)

        # Ensure the input spectrum for classification has the target pipeline length
        if len(denoised_spectrum) != TARGET_SPEC_LENGTH:
             logger.error(f"Input spectrum for classification has incorrect length: {len(denoised_spectrum)}. Expected: {TARGET_SPEC_LENGTH}.")
             raise HTTPException(status_code=400, detail=f"Classification input spectrum has incorrect length ({len(denoised_spectrum)}). It must be exactly {TARGET_SPEC_LENGTH} points after denoising.")

        # SynCleanSet is loaded at startup and its shape/length checked against TARGET_SPEC_LENGTH.
        # Its shape is (TotalSamples, TARGET_SPEC_LENGTH).
        total_synset_samples = SynCleanSet.shape[0]
        # synset_spectrum_length is TARGET_SPEC_LENGTH based on startup check
        synset_spectrum_length = SynCleanSet.shape[1]

        expected_samples_per_material = NumCleanSpec

        best_score = -1
        best_plastic_type = None
        best_clean_spectrum = None # To store the best matching average clean spectrum

        # Iterate through each material defined in NameList
        for i, name in enumerate(NameList):
            start_index = i * expected_samples_per_material
            end_index = (i + 1) * expected_samples_per_material

            if start_index >= total_synset_samples:
                 logger.warning(f"Skipping material {name} (index {i}): Start index {start_index} is beyond total samples in SynCleanSet ({total_synset_samples}).")
                 break

            current_end_index = min(end_index, total_synset_samples)

            # Extract the spectra for the current material using numpy slicing.
            # Both denoised_spectrum and average_clean_spectrum will have length TARGET_SPEC_LENGTH (6501)
            spectra_for_material = SynCleanSet[start_index:current_end_index, :]

            if spectra_for_material.shape[0] == 0:
                 logger.warning(f"No samples extracted for material {name} using indices {start_index}:{current_end_index}. Skipping material.")
                 continue

            # Calculate the average spectrum for this material by taking the mean along axis 0 (across samples)
            average_clean_spectrum = np.mean(spectra_for_material, axis=0)

            # Both denoised_spectrum and average_clean_spectrum now have length TARGET_SPEC_LENGTH (6501).
            # Pearson correlation can be calculated.
            if np.std(denoised_spectrum) == 0 or np.std(average_clean_spectrum) == 0:
                 current_score = 0.0 # Correlation is undefined or zero if data is constant
                 if np.std(denoised_spectrum) == 0: logger.warning("Input spectrum is constant. Pearson correlation is 0.")
                 if np.std(average_clean_spectrum) == 0: logger.warning(f"Average clean spectrum for {name} is constant. Pearson correlation is 0.")
            else:
                 try:
                    current_score, _ = pearsonr(denoised_spectrum, average_clean_spectrum)
                    if np.isnan(current_score):
                         current_score = 0.0
                         logger.warning(f"Pearson correlation calculated as NaN for {name}. Setting to 0.")
                 except Exception as pearson_e:
                    logger.error(f"Error calculating Pearson correlation for {name}: {pearson_e}")
                    current_score = 0.0

            if current_score > best_score:
                best_score = current_score
                best_plastic_type = name
                best_clean_spectrum = average_clean_spectrum.tolist()

        # --- Handle case where no valid match was found ---
        if best_plastic_type is None or best_score <= 0:
             # If no material was found with a positive score, classify as Unknown
             logger.warning(f"Classification process completed. Best Pearson correlation was {best_score:.4f}. Classified as Unknown.")
             # Decide if you return the best (non-positive) match spectrum or an empty list
             return JSONResponse(content={
                 'plastic_type': "Unknown",
                 'clean_spectrum': best_clean_spectrum if best_plastic_type is not None else [], # Return best non-positive match spectrum if one was found, else empty
                 'pearson_correlation': float(best_score) if best_plastic_type is not None else 0.0,
                 'model_used': model_name,
                 'message': f"Best correlation score ({best_score:.4f}) was not positive." if best_plastic_type is not None else "Classification could not identify a plastic type."
             })


        # --- Return the classification result for the best positive match ---
        return JSONResponse(content={
            'plastic_type': best_plastic_type,
            'clean_spectrum': best_clean_spectrum,
            'pearson_correlation': float(best_score), # Ensure it's a standard float for JSON
            'model_used': model_name
        })

    except (ValueError, json.JSONDecodeError) as e:
         logger.error(f"Classification input data error: {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")
    except HTTPException as e:
         raise e
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during classification: {str(e)}")