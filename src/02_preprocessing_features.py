"""
Heart Sound Preprocessing and Feature Extraction
===============================================

Professional preprocessing pipeline for heart sound classification.
Includes advanced feature extraction techniques optimized for biomedical audio.

Features:
- Audio preprocessing and normalization
- Multi-scale feature extraction (MFCCs, Spectrograms, Chroma)
- Heart cycle segmentation
- Data augmentation for training
- Quality assessment and filtering

Author: ML Engineer for Heart Sound Classification
Version: 1.0
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeartSoundPreprocessor:
    """
    Advanced preprocessing pipeline for heart sound classification.
    Optimized for biomedical audio with clinical-grade feature extraction.
    """
    
    def __init__(self, 
                 target_sr: int = 2000,
                 segment_duration: float = 5.0,
                 overlap_ratio: float = 0.5):
        """
        Initialize the preprocessor with optimal parameters for heart sounds.
        
        Args:
            target_sr: Target sampling rate (2000 Hz optimal for heart sounds)
            segment_duration: Duration of each segment in seconds
            overlap_ratio: Overlap ratio for segmentation
        """
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.overlap_ratio = overlap_ratio
        self.segment_samples = int(target_sr * segment_duration)
        
        # Feature extraction parameters
        self.mfcc_params = {
            'n_mfcc': 13,
            'n_fft': 512,
            'hop_length': 256,
            'n_mels': 40
        }
        
        self.spectral_params = {
            'n_fft': 512,
            'hop_length': 256,
            'n_mels': 64
        }
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file with error handling.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=None)
            
            # Resample if necessary
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize audio
            audio = self.normalize_audio(audio)
            
            logger.debug(f"Loaded {file_path}: {len(audio)} samples at {self.target_sr} Hz")
            return audio, self.target_sr
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return np.array([]), 0
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio signal with clinical-grade preprocessing.
        
        Args:
            audio: Raw audio signal
            
        Returns:
            Normalized audio signal
        """
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Apply high-pass filter to remove low-frequency noise
        sos = signal.butter(4, 40, btype='high', fs=self.target_sr, output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # Apply low-pass filter to remove high-frequency noise
        sos = signal.butter(4, 800, btype='low', fs=self.target_sr, output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio into overlapping windows for analysis.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            List of audio segments
        """
        segments = []
        step_size = int(self.segment_samples * (1 - self.overlap_ratio))
        
        for start in range(0, len(audio) - self.segment_samples + 1, step_size):
            segment = audio[start:start + self.segment_samples]
            
            # Only keep segments with sufficient energy
            if np.std(segment) > 0.01:  # Threshold for meaningful signal
                segments.append(segment)
        
        return segments
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features optimized for heart sounds.
        
        Args:
            audio: Audio segment
            
        Returns:
            MFCC feature vector
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.target_sr,
                **self.mfcc_params
            )
            
            # Calculate statistics across time
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
            
            # Combine all MFCC features
            features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta, mfcc_delta2])
            
            return features
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            return np.zeros(self.mfcc_params['n_mfcc'] * 4)
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features relevant for heart sound analysis.
        
        Args:
            audio: Audio segment
            
        Returns:
            Spectral feature vector
        """
        # Compute safe spectral contrast parameters and expected length locally
        nyquist = self.target_sr / 2.0
        fmin = max(20.0, min(200.0, nyquist / 32.0))
        max_n_bands = int(np.floor(np.log2(nyquist / fmin))) if nyquist > fmin else 1
        n_bands = int(np.clip(max_n_bands, 1, 6))  # librosa default is 6
        expected_length = 5 + (n_bands + 1) + 12  # base(5) + contrast + chroma(12)
        try:
            # Spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.target_sr
            ))
            
            # Spectral rolloff
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio, sr=self.target_sr
            ))
            
            # Spectral bandwidth
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=audio, sr=self.target_sr
            ))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Root mean square energy
            rms = np.mean(librosa.feature.rms(y=audio))
            
            # Spectral contrast (ensure bands within Nyquist limit)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(
                y=audio, sr=self.target_sr, fmin=fmin, n_bands=n_bands
            ), axis=1)
            
            # Chroma features
            chroma = np.mean(librosa.feature.chroma_stft(
                y=audio, sr=self.target_sr
            ), axis=1)
            
            # Combine all spectral features
            features = np.concatenate([
                [spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr, rms],
                spectral_contrast,
                chroma
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return np.zeros(expected_length)  # Match expected spectral feature size
    
    def extract_temporal_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract temporal features specific to heart sound patterns.
        
        Args:
            audio: Audio segment
            
        Returns:
            Temporal feature vector
        """
        try:
            # Basic statistical features
            mean_val = np.mean(audio)
            std_val = np.std(audio)
            skewness = skew(audio)
            kurt = kurtosis(audio)
            
            # Energy-based features
            energy = np.sum(audio ** 2)
            power = energy / len(audio)
            
            # Peak detection for heart rate estimation
            peaks, _ = signal.find_peaks(np.abs(audio), 
                                       height=0.1 * np.max(np.abs(audio)),
                                       distance=int(0.4 * self.target_sr))  # Min 150 BPM
            
            heart_rate_estimate = len(peaks) * 60 / self.segment_duration
            
            # Envelope features
            envelope = np.abs(signal.hilbert(audio))
            envelope_mean = np.mean(envelope)
            envelope_std = np.std(envelope)
            
            features = np.array([
                mean_val, std_val, skewness, kurt,
                energy, power, heart_rate_estimate,
                envelope_mean, envelope_std
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            return np.zeros(9)
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature set for heart sound classification.
        
        Args:
            audio: Audio segment
            
        Returns:
            Complete feature vector
        """
        # Extract different feature types
        mfcc_features = self.extract_mfcc_features(audio)
        spectral_features = self.extract_spectral_features(audio)
        temporal_features = self.extract_temporal_features(audio)
        
        # Combine all features
        all_features = np.concatenate([
            mfcc_features,
            spectral_features, 
            temporal_features
        ])
        
        return all_features
    
    def assess_audio_quality(self, audio: np.ndarray) -> float:
        """
        Assess the quality of audio recording for filtering.
        
        Args:
            audio: Audio segment
            
        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio ** 2)
            noise_estimate = np.std(audio[:int(0.1 * len(audio))])  # First 10% as noise
            snr = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))
            
            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            
            # Clipping detection
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            
            # Silence detection
            silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
            
            # Combine quality metrics
            quality_score = (
                min(snr / 20, 1.0) * 0.4 +  # SNR component
                min(dynamic_range / 0.5, 1.0) * 0.3 +  # Dynamic range component
                (1 - clipping_ratio) * 0.2 +  # Anti-clipping component
                (1 - min(silence_ratio / 0.5, 1.0)) * 0.1  # Anti-silence component
            )
            
            return max(0, min(1, quality_score))
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5  # Default moderate quality
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str = 'all') -> List[np.ndarray]:
        """
        Apply data augmentation techniques for heart sound data.
        
        Args:
            audio: Original audio segment
            augmentation_type: Type of augmentation ('noise', 'stretch', 'pitch', 'all')
            
        Returns:
            List of augmented audio segments
        """
        augmented_samples = [audio]  # Include original
        
        try:
            if augmentation_type in ['noise', 'all']:
                # Add controlled noise
                noise_level = 0.005 * np.random.uniform(0.5, 2.0)
                noise = np.random.normal(0, noise_level, len(audio))
                augmented_samples.append(audio + noise)
            
            if augmentation_type in ['stretch', 'all']:
                # Time stretching (preserve heart rhythm characteristics)
                stretch_factor = np.random.uniform(0.9, 1.1)
                stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
                if len(stretched) >= self.segment_samples:
                    augmented_samples.append(stretched[:self.segment_samples])
                else:
                    # Pad if too short
                    padded = np.pad(stretched, (0, self.segment_samples - len(stretched)), 'constant')
                    augmented_samples.append(padded)
            
            if augmentation_type in ['pitch', 'all']:
                # Slight pitch shifting (within physiological range)
                pitch_shift = np.random.uniform(-2, 2)  # semitones
                pitched = librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=pitch_shift)
                augmented_samples.append(pitched)
                
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
        
        return augmented_samples

class FeatureExtractor:
    """
    High-level feature extraction interface for heart sound classification.
    """
    
    def __init__(self, preprocessor: HeartSoundPreprocessor):
        """
        Initialize feature extractor.
        
        Args:
            preprocessor: Configured preprocessor instance
        """
        self.preprocessor = preprocessor
    
    def process_physionet_dataset(self, 
                                 base_dir: str = ".",
                                 output_file: str = 'physionet_heart_features.csv',
                                 augment: bool = True,
                                 max_files_per_db: int = None) -> pd.DataFrame:
        """
        Process PhysioNet Challenge 2016 dataset and extract features.
        
        Args:
            base_dir: Base directory containing training-a through training-f
            output_file: Output file for features
            augment: Whether to apply data augmentation
            max_files_per_db: Maximum files to process per database (for testing)
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Processing PhysioNet Challenge 2016 dataset from {base_dir}")
        
        # Initialize feature storage
        all_features = []
        all_labels = []
        all_filenames = []
        all_quality_scores = []
        all_databases = []
        
        # Process each training database
        training_dirs = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
        
        for db_name in training_dirs:
            db_path = Path(base_dir) / db_name
            if not db_path.exists():
                logger.warning(f"Database {db_name} not found at {db_path}")
                continue
                
            # Load reference file for this database
            ref_file = db_path / 'REFERENCE.csv'
            if not ref_file.exists():
                logger.warning(f"REFERENCE.csv not found in {db_name}")
                continue
                
            logger.info(f"Processing database: {db_name}")
            
            # Load labels
            try:
                ref_df = pd.read_csv(ref_file, header=None, names=['filename', 'label'])
                logger.info(f"Found {len(ref_df)} files in {db_name}")
            except Exception as e:
                logger.error(f"Failed to load REFERENCE.csv from {db_name}: {e}")
                continue
            
            # Process files in this database
            files_processed = 0
            for idx, row in ref_df.iterrows():
                if max_files_per_db and files_processed >= max_files_per_db:
                    break
                    
                filename = row['filename']
                label_code = row['label']
                
                # Convert PhysioNet labels (handle string or numeric)
                # PhysioNet REFERENCE.csv typically uses 'normal'/'abnormal'.
                label_str = str(label_code).strip().lower()
                if label_str in ['abnormal', '1', 'a', 'abn']:
                    label = 'abnormal'
                else:
                    label = 'normal'
                
                # Construct file path
                wav_file = db_path / f"{filename}.wav"
                
                if not wav_file.exists():
                    logger.warning(f"Audio file not found: {wav_file}")
                    continue
                
                try:
                    # Load and preprocess audio
                    audio, sr = self.preprocessor.load_audio(str(wav_file))
                    
                    if len(audio) == 0:
                        continue
                    
                    # Segment audio
                    segments = self.preprocessor.segment_audio(audio)
                    
                    # Process each segment
                    for seg_idx, segment in enumerate(segments):
                        # Assess quality
                        quality = self.preprocessor.assess_audio_quality(segment)
                        
                        # Skip low-quality segments
                        if quality < 0.3:
                            continue
                        
                        # Extract features
                        features = self.preprocessor.extract_all_features(segment)
                        
                        # Store results
                        all_features.append(features)
                        all_labels.append(label)
                        all_filenames.append(f"{filename}_seg_{seg_idx}")
                        all_quality_scores.append(quality)
                        all_databases.append(db_name)
                        
                        # Apply augmentation if requested
                        if augment and quality > 0.7:  # Only augment high-quality segments
                            augmented_segments = self.preprocessor.augment_audio(segment)
                            
                            for aug_idx, aug_segment in enumerate(augmented_segments[1:]):  # Skip original
                                aug_features = self.preprocessor.extract_all_features(aug_segment)
                                all_features.append(aug_features)
                                all_labels.append(label)
                                all_filenames.append(f"{filename}_seg_{seg_idx}_aug_{aug_idx}")
                                all_quality_scores.append(quality * 0.9)  # Slightly lower quality for augmented
                                all_databases.append(db_name)
                    
                    files_processed += 1
                    if files_processed % 50 == 0:
                        logger.info(f"Processed {files_processed} files from {db_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
                    continue
        
        # Create DataFrame from all processed data
        if all_features:
            feature_array = np.array(all_features)
            
            # Create column names (match actual extracted sizes)
            n_mfcc = self.preprocessor.mfcc_params['n_mfcc']
            mfcc_cols = [f'mfcc_{i}' for i in range(n_mfcc * 4)]  # mean, std, delta, delta2
            base_spectral_cols = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zcr', 'rms']
            chroma_cols = [f'chroma_{i}' for i in range(12)]
            temporal_cols = ['mean', 'std', 'skewness', 'kurtosis', 'energy', 'power', 
                           'heart_rate_est', 'envelope_mean', 'envelope_std']
            
            # Determine spectral_contrast vector length from total feature length
            expected_fixed = len(mfcc_cols) + len(base_spectral_cols) + len(chroma_cols) + len(temporal_cols)
            contrast_len = max(0, feature_array.shape[1] - expected_fixed)
            spectral_contrast_cols = [f'spectral_contrast_{i}' for i in range(contrast_len)]
            
            all_cols = mfcc_cols + base_spectral_cols + spectral_contrast_cols + chroma_cols + temporal_cols
            
            # Create DataFrame with exact matching columns
            df = pd.DataFrame(feature_array, columns=all_cols)
            df['label'] = all_labels
            df['filename'] = all_filenames
            df['quality_score'] = all_quality_scores
            df['database'] = all_databases
            
            # Save to file
            df.to_csv(output_file, index=False)
            logger.info(f"Features saved to {output_file}")
            logger.info(f"PhysioNet dataset summary: {len(df)} samples")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
            logger.info(f"Database distribution: {df['database'].value_counts().to_dict()}")
            logger.info(f"Average quality score: {df['quality_score'].mean():.3f}")
            
            return df
        
        else:
            logger.error("No features extracted from PhysioNet dataset")
            return pd.DataFrame()
    
    def process_dataset(self, 
                       data_dir: str, 
                       metadata_file: str,
                       output_file: str = 'heart_sound_features.csv',
                       augment: bool = True) -> pd.DataFrame:
        """
        Legacy method for backward compatibility.
        For PhysioNet dataset, use process_physionet_dataset instead.
        """
        logger.warning("Using legacy process_dataset method. Consider using process_physionet_dataset for PhysioNet data.")
        return self.process_physionet_dataset(data_dir, output_file, augment)

def main():
    """
    Example usage of the preprocessing pipeline for PhysioNet Challenge 2016 dataset.
    """
    # Initialize preprocessor
    preprocessor = HeartSoundPreprocessor()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(preprocessor)
    
    # Process PhysioNet Challenge 2016 dataset
    print("Processing PhysioNet Challenge 2016 Heart Sound Dataset...")
    print("This will process all training databases (training-a through training-f)")
    print("Expected processing time: 10-30 minutes depending on your system")
    
    # Extract features from PhysioNet dataset
    features_df = extractor.process_physionet_dataset(
        base_dir=".",  # Current directory should contain training-a, training-b, etc.
        output_file="physionet_heart_features.csv",
        augment=True,
        max_files_per_db=100  # Process first 100 files per database for testing (remove for full dataset)
    )
    
    if not features_df.empty:
        print(f"\nFeature extraction completed successfully!")
        print(f"Dataset shape: {features_df.shape}")
        print(f"Features saved to: physionet_heart_features.csv")
        
        # Display comprehensive statistics
        print("\n" + "="*60)
        print("PHYSIONET DATASET SUMMARY")
        print("="*60)
        print(f"Total samples: {len(features_df):,}")
        print(f"Total features: {features_df.shape[1] - 4}")  # Excluding label, filename, quality, database
        
        print(f"\nLabel Distribution:")
        label_counts = features_df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(features_df)) * 100
            print(f"  {label.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nDatabase Distribution:")
        db_counts = features_df['database'].value_counts()
        for db, count in db_counts.items():
            percentage = (count / len(features_df)) * 100
            print(f"  {db}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nQuality Score Statistics:")
        quality_stats = features_df['quality_score'].describe()
        print(f"  Mean: {quality_stats['mean']:.3f}")
        print(f"  Std:  {quality_stats['std']:.3f}")
        print(f"  Min:  {quality_stats['min']:.3f}")
        print(f"  Max:  {quality_stats['max']:.3f}")
        
        # Check for class imbalance
        normal_count = label_counts.get('normal', 0)
        abnormal_count = label_counts.get('abnormal', 0)
        if normal_count > 0 and abnormal_count > 0:
            imbalance_ratio = max(normal_count, abnormal_count) / min(normal_count, abnormal_count)
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 3:
                print("  ⚠️  Significant class imbalance detected - consider using stratified sampling")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run model training: python 03_model_training.py")
        print("2. The feature file 'physionet_heart_features.csv' is ready for ML training")
        print("3. Consider removing max_files_per_db limit for full dataset processing")
        print("="*60)
        
    else:
        print("❌ Feature extraction failed. Please check:")
        print("1. PhysioNet dataset is properly downloaded")
        print("2. Directory structure: training-a/, training-b/, etc. exist")
        print("3. Each training directory contains REFERENCE.csv and .wav files")
        print("4. Run 01_data_downloader.py first if dataset is missing")
    print("Ready for model training with extracted features.")

if __name__ == "__main__":
    main()
