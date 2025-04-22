import unittest
import numpy as np
import pyopencl as cl
from mandelbrot_opencl import mandelbrot_gpu, create_mesh

class TestMandelbrotOpenCL(unittest.TestCase):
    
    def test_basic_mandelbrot_computation(self):
        """Test basic computation of Mandelbrot set - verifies if computation runs without errors
        and returns an array of expected shape with valid values."""
        # small test grid
        C = create_mesh(-2.0, 1.0, -1.5, 1.5, 100, 100)
        
        # run on GPU
        platform = "GPU"
        max_iters = 100
        threshold = 2.0
        
        try:
            result = mandelbrot_gpu(C, np.int32(max_iters), np.float32(threshold), platform)
            
            # check shape and type
            self.assertEqual(result.shape, (100, 100))
            self.assertEqual(result.dtype, np.int32)
            
            # check that all fields are between 0 and max_iters
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= max_iters))
            
            # check that not all pixels are at max_iters
            self.assertTrue(np.any(result == max_iters))
            self.assertTrue(np.any(result < max_iters))
            
        except cl.LogicError as e:
            self.skipTest(f"OpenCL error: {e}")
    
    def test_known_points(self):
        """Test specific points with known behavior in the Mandelbrot set."""
        # create a grid with specific points
        # 0+0j is in the set
        # 1+0j is not in the set
        # -1+0j is in the set
        # -2+0j is not in the set
        test_points = np.array([[0+0j, 1+0j], [-1+0j, -2+0j]], dtype=np.complex64)
        
        platform = "CPU"
        max_iters = 100
        threshold = 2.0
        
        try:
            result = mandelbrot_gpu(test_points, np.int32(max_iters), np.float32(threshold), platform)
            print(result)
            
            # 0+0j should not escape
            self.assertEqual(result[0, 0], max_iters)
            
            # 1+0j should escape
            self.assertLess(result[0, 1], 5)  # less than 5 in case it doesn't diverge immediately
            
            # -1+0j should not escape
            self.assertEqual(result[1, 0], max_iters)
            
            # -2+0j should escape
            self.assertLess(result[1, 1], 5)
            
        except cl.LogicError as e:
            self.skipTest(f"OpenCL error: {e}")
    
    def test_grid_size_parameter(self):
        """Test if the grid_size affects results."""
        # small test grid
        C = create_mesh(-2.0, 1.0, -1.5, 1.5, 256, 256)
        
        # run on GPU
        platform = "GPU"
        max_iters = 50
        threshold = 2.0
        
        try:
            # test with grid size = None. The default
            result1 = mandelbrot_gpu(C, np.int32(max_iters), np.float32(threshold), platform)
            
            # test with given grid_size
            grid_size = (16,)  # tuple with single value
            result2 = mandelbrot_gpu(C, np.int32(max_iters), np.float32(threshold), platform, grid_size)
            
            # check that results are equal
            np.testing.assert_array_equal(result1, result2)
            
            # check that the shape is correct
            self.assertEqual(result1.shape, (256, 256))
            self.assertEqual(result2.shape, (256, 256))
            
        except cl.LogicError as e:
            self.skipTest(f"OpenCL error: {e}")

if __name__ == '__main__':
    unittest.main()