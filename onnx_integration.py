#!/usr/bin/env python
"""
ONNX Runtime Integration for Enhanced Deep Learning Pattern Recognition.

This module provides functionality to export PyTorch models to ONNX format
and optimize inference using ONNX Runtime.
"""

import os
import time
import logging
import numpy as np
import torch
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('onnx_integration')

class ONNXModelExporter:
    """Class for exporting PyTorch models to ONNX format."""
    
    def __init__(self, output_dir: str = 'models'):
        """Initialize the ONNX model exporter.
        
        Args:
            output_dir: Directory to save ONNX models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized ONNXModelExporter with output_dir={output_dir}")
    
    def export_model(self, 
                    model: torch.nn.Module, 
                    model_name: str, 
                    input_shape: Tuple[int, ...],
                    dynamic_axes: Optional[Dict] = None,
                    opset_version: int = 12) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            model_name: Name for the exported model
            input_shape: Shape of the input tensor
            dynamic_axes: Dynamic axes for variable length inputs
            opset_version: ONNX opset version
            
        Returns:
            Path to the exported ONNX model
        """
        # Ensure model is in evaluation mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Define output path
        onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")
        
        # Set dynamic axes if not provided
        if dynamic_axes is None and len(input_shape) > 2:
            # Default dynamic axes for sequence models (batch_size, seq_len, features)
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        
        # Export the model
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"Model exported to {onnx_path}")
            
            # Verify the model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified successfully")
            
            return onnx_path
        
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def optimize_model(self, onnx_path: str, optimization_level: int = 99) -> str:
        """Optimize ONNX model for inference.
        
        Args:
            onnx_path: Path to the ONNX model
            optimization_level: ONNX optimization level (0-99)
            
        Returns:
            Path to the optimized ONNX model
        """
        try:
            import onnxoptimizer
            
            # Load the model
            onnx_model = onnx.load(onnx_path)
            
            # Optimize the model
            optimized_model = onnxoptimizer.optimize(onnx_model)
            
            # Save the optimized model
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            logger.info(f"Model optimized and saved to {optimized_path}")
            return optimized_path
        
        except ImportError:
            logger.warning("onnxoptimizer not installed. Skipping optimization.")
            return onnx_path
        
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return onnx_path


class ONNXRuntimeInference:
    """Class for performing inference using ONNX Runtime."""
    
    def __init__(self, 
                onnx_path: str, 
                providers: List[str] = None,
                enable_profiling: bool = False):
        """Initialize the ONNX Runtime inference.
        
        Args:
            onnx_path: Path to the ONNX model
            providers: List of execution providers (e.g., 'CPUExecutionProvider', 'CUDAExecutionProvider')
            enable_profiling: Whether to enable profiling
        """
        # Set default providers if not provided
        if providers is None:
            providers = ['CPUExecutionProvider']
            
            # Add CUDA provider if available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
        
        # Create session options
        session_options = ort.SessionOptions()
        
        # Enable profiling if requested
        if enable_profiling:
            session_options.enable_profiling = True
            session_options.profile_file_prefix = "onnx_profile"
        
        # Set graph optimization level
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create inference session
        try:
            self.session = ort.InferenceSession(
                onnx_path, 
                sess_options=session_options,
                providers=providers
            )
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input shape
            self.input_shape = self.session.get_inputs()[0].shape
            
            logger.info(f"Initialized ONNXRuntimeInference with model {onnx_path}")
            logger.info(f"Using providers: {providers}")
            logger.info(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Perform inference using ONNX Runtime.
        
        Args:
            input_data: Input data as numpy array
            
        Returns:
            Model predictions as numpy array
        """
        try:
            # Ensure input data has the correct shape
            if len(input_data.shape) == 2 and len(self.input_shape) == 3:
                # Add batch dimension
                input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: input_data.astype(np.float32)}
            )
            
            return outputs[0]
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def benchmark(self, 
                 input_shape: Tuple[int, ...], 
                 num_iterations: int = 100,
                 warmup_iterations: int = 10) -> Dict:
        """Benchmark inference performance.
        
        Args:
            input_shape: Shape of the input tensor
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        # Create random input data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            self.predict(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.predict(input_data)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        results = {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'num_iterations': num_iterations,
            'input_shape': input_shape
        }
        
        logger.info(f"Benchmark results: avg_time={avg_time:.6f}s, throughput={throughput:.2f} inferences/s")
        return results


class ONNXModelManager:
    """Class for managing ONNX models and inference."""
    
    def __init__(self, model_dir: str = 'models'):
        """Initialize the ONNX model manager.
        
        Args:
            model_dir: Directory for ONNX models
        """
        self.model_dir = model_dir
        self.exporter = ONNXModelExporter(output_dir=model_dir)
        self.inference_sessions = {}
        
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Initialized ONNXModelManager with model_dir={model_dir}")
    
    def export_and_optimize(self, 
                           model: torch.nn.Module, 
                           model_name: str, 
                           input_shape: Tuple[int, ...]) -> str:
        """Export and optimize a PyTorch model.
        
        Args:
            model: PyTorch model to export
            model_name: Name for the exported model
            input_shape: Shape of the input tensor
            
        Returns:
            Path to the optimized ONNX model
        """
        # Export the model
        onnx_path = self.exporter.export_model(model, model_name, input_shape)
        
        # Optimize the model
        optimized_path = self.exporter.optimize_model(onnx_path)
        
        return optimized_path
    
    def load_model(self, 
                  model_name: str, 
                  providers: List[str] = None,
                  enable_profiling: bool = False) -> str:
        """Load an ONNX model for inference.
        
        Args:
            model_name: Name of the model to load
            providers: List of execution providers
            enable_profiling: Whether to enable profiling
            
        Returns:
            Model ID for inference
        """
        # Check if optimized model exists
        optimized_path = os.path.join(self.model_dir, f"{model_name}_optimized.onnx")
        if os.path.exists(optimized_path):
            model_path = optimized_path
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.onnx")
            
            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_name} not found in {self.model_dir}")
        
        # Create inference session
        model_id = f"{model_name}_{int(time.time())}"
        self.inference_sessions[model_id] = ONNXRuntimeInference(
            model_path, 
            providers=providers,
            enable_profiling=enable_profiling
        )
        
        logger.info(f"Model {model_name} loaded with ID {model_id}")
        return model_id
    
    def predict(self, model_id: str, input_data: np.ndarray) -> np.ndarray:
        """Perform inference using a loaded model.
        
        Args:
            model_id: ID of the loaded model
            input_data: Input data as numpy array
            
        Returns:
            Model predictions as numpy array
        """
        if model_id not in self.inference_sessions:
            raise ValueError(f"Model ID {model_id} not found")
        
        return self.inference_sessions[model_id].predict(input_data)
    
    def benchmark_model(self, 
                       model_id: str, 
                       input_shape: Tuple[int, ...],
                       num_iterations: int = 100) -> Dict:
        """Benchmark a loaded model.
        
        Args:
            model_id: ID of the loaded model
            input_shape: Shape of the input tensor
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        if model_id not in self.inference_sessions:
            raise ValueError(f"Model ID {model_id} not found")
        
        return self.inference_sessions[model_id].benchmark(
            input_shape, 
            num_iterations=num_iterations
        )
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory.
        
        Args:
            model_id: ID of the model to unload
        """
        if model_id in self.inference_sessions:
            del self.inference_sessions[model_id]
            logger.info(f"Model {model_id} unloaded")
        else:
            logger.warning(f"Model ID {model_id} not found")


if __name__ == "__main__":
    import argparse
    from enhanced_dl_model_fixed import EnhancedPatternRecognitionModel
    
    parser = argparse.ArgumentParser(description='ONNX Runtime Integration')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory for ONNX models')
    parser.add_argument('--input_dim', type=int, default=9, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for benchmarking')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for benchmarking')
    args = parser.parse_args()
    
    # Create model manager
    manager = ONNXModelManager(model_dir=args.model_dir)
    
    # Create PyTorch model
    model = EnhancedPatternRecognitionModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    
    # Export and optimize model
    optimized_path = manager.export_and_optimize(
        model,
        "pattern_recognition",
        (args.batch_size, args.seq_length, args.input_dim)
    )
    
    # Load model for inference
    model_id = manager.load_model("pattern_recognition")
    
    # Benchmark model
    results = manager.benchmark_model(
        model_id,
        (args.batch_size, args.seq_length, args.input_dim)
    )
    
    print(f"ONNX Runtime Benchmark Results:")
    print(f"Average Inference Time: {results['avg_inference_time']:.6f} seconds")
    print(f"Throughput: {results['throughput']:.2f} inferences/second")
