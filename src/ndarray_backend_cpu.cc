#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits> 
namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  int numDimensions = 0;
  for (int i = 0; i < shape.size(); ++i) {
        // std::cout << "Element " << i << ": " << strides[i] << "\t";
        //std::cout << "shapes " << i << ": " << shape[i] << "\t";
        numDimensions= numDimensions+1;
      }
  std::vector<int> indices(numDimensions);
  // Initialize the indices
    for (int i = 0; i < numDimensions; ++i) {
        indices[i] = 0;  
    }
  //std::cout<<"first";
  for(int j=0; j <out->size; j++)
  {
    int cur_index = 0;
    // for (int i = 0; i <numDimensions; i++){
    //   std::cout<<indices[i]<<"\t";
    // } 
    for (int i = 0; i < numDimensions; i++) 
    {
      cur_index = cur_index+ indices[i]*strides[i];
    }
    int carry=0;
    indices[numDimensions-1]= indices[numDimensions-1] + 1;

    for (int i = numDimensions-1; i >= 0; i--) 
    {
      if (indices[i] == shape[i]) {
            indices[i] = 0;
            carry = 1; 
        } else {
            indices[i] = carry + indices[i];
            if(indices[i]==shape[i])
            {
              indices[i] =0;
              carry=1;
            }
            else{
            carry = 0;}
        }
    }
    // for (int i = 0; i <numDimensions; i++){
    //   std::cout<<indices[i]<<"\t";
    // } 
    //std::cout<<std::endl;
    out->ptr[j] = a.ptr[offset+cur_index];  
    }
    return;
  }
  /// END SOLUTION


void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  int numDimensions = 0;
  for (int i = 0; i < shape.size(); ++i) {
        // std::cout << "Element " << i << ": " << strides[i] << "\t";
        //std::cout << "shapes " << i << ": " << shape[i] << "\t";
        numDimensions= numDimensions+1;
      }
  std::vector<int> indices(numDimensions);
  // Initialize the indices
    for (int i = 0; i < numDimensions; ++i) {
        indices[i] = 0;  
    }
  //std::cout<<"first";
  for(int j=0; j <a.size; j++)
  {
    int cur_index = 0;
    // for (int i = 0; i <numDimensions; i++){
    //   std::cout<<indices[i]<<"\t";
    // } 
    for (int i = 0; i < numDimensions; i++) 
    {
      cur_index = cur_index+ indices[i]*strides[i];
    }
    int carry=0;
    indices[numDimensions-1]= indices[numDimensions-1] + 1;

    for (int i = numDimensions-1; i >= 0; i--) 
    {
      if (indices[i] == shape[i]) {
            indices[i] = 0;
            carry = 1; 
        } else {
            indices[i] = carry + indices[i];
            if(indices[i]==shape[i])
            {
              indices[i] =0;
              carry=1;
            }
            else{
            carry = 0;}
        }
    }
    // for (int i = 0; i <numDimensions; i++){
    //   std::cout<<indices[i]<<"\t";
    // } 
    //std::cout<<std::endl;
    out->ptr[offset+cur_index] = a.ptr[j];  
    }
    return;
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  int numDimensions = 0;
  for (int i = 0; i < shape.size(); ++i) {
        // std::cout << "Element " << i << ": " << strides[i] << "\t";
        //std::cout << "shapes " << i << ": " << shape[i] << "\t";
        numDimensions= numDimensions+1;
      }
  std::vector<int> indices(numDimensions);
  // Initialize the indices
    for (int i = 0; i < numDimensions; ++i) {
        indices[i] = 0;  
    }
  //std::cout<<"first";
  for(int j=0; j <out->size; j++)
  {
    int cur_index = 0;
    // for (int i = 0; i <numDimensions; i++){
    //   std::cout<<indices[i]<<"\t";
    // } 
    for (int i = 0; i < numDimensions; i++) 
    {
      cur_index = cur_index+ indices[i]*strides[i];
    }
    int carry=0;
    indices[numDimensions-1]= indices[numDimensions-1] + 1;

    for (int i = numDimensions-1; i >= 0; i--) 
    {
      if (indices[i] == shape[i]) {
            indices[i] = 0;
            carry = 1; 
        } else {
            indices[i] = carry + indices[i];
            if(indices[i]==shape[i])
            {
              indices[i] =0;
              carry=1;
            }
            else{
            carry = 0;}
        }
    }
    // for (int i = 0; i <numDimensions; i++){
    //   std::cout<<indices[i]<<"\t";
    // } 
    //std::cout<<std::endl;
    out->ptr[offset+cur_index] = val;  
    }
    return;
  //assert(false && "Not Implemented");
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entry into the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entry into the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entry into the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val); 
  }
}
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entry into the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    if (a.ptr[i]>b.ptr[i])
    {
      out->ptr[i] = a.ptr[i];
    }
    else
    {
      out->ptr[i] = b.ptr[i];
    }
  
  }
}
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entry into the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    if (a.ptr[i]>val)
    {
      out->ptr[i] = a.ptr[i];
    }
    else
    {
      out->ptr[i] = val;
    }
  }
}
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] == b.ptr[i]) ? 1.0f : 0.0f;
    }
}
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] == val) ? 1.0f : 0.0f;
    }
}
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] >= b.ptr[i]) ? 1.0f : 0.0f;
    }
}
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = (a.ptr[i] >= val) ? 1.0f : 0.0f;
    }
}
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::log(a.ptr[i]);
    }
}
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::exp(a.ptr[i]);
    }
}
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = std::tanh(a.ptr[i]);
    }
}
/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

 
  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t k = 0; k < p; k++) {
         out->ptr[i * p + k] = 0;
        for (uint32_t j = 0; j < n; j++) {
            out->ptr[i * p + k] = out->ptr[i * p + k]+ a.ptr[i * n + j] * b.ptr[j * p + k];
        }
        
    }
}
  /// END SOLUTION
}


void LUDecomposition(const AlignedArray& a, AlignedArray* L, AlignedArray* U, uint32_t n) {

	// Set lower triangular diagonal to 1
	for (int k = 0; k < n; ++k) {
		L->ptr[k*n+k] = 1;
	}
	for (int k = 0; k < n; ++k) {
        // U matrix (upper triangular)
        for (int j = k; j < n; ++j) {
            U->ptr[k*n+j] = a.ptr[k*n+j];
            for (int i = 0; i < k; ++i) {
                U->ptr[k*n+j] -= L->ptr[k*n+i] * U->ptr[i*n+j];
            }
        }

        // L matrix (lower triangular)
        for (int i = k + 1; i < n; ++i) {
            L->ptr[i*n+k] = a.ptr[i*n+k];
            for (int j = 0; j < k; ++j) {
                L->ptr[i*n+k] -= L->ptr[i*n+j] * U->ptr[j*n+k];
            }
            L->ptr[i*n+k] /= U->ptr[k*n+k];
        }
    }
}
void GaussNewton(const AlignedArray& a, AlignedArray* L, AlignedArray* U, uint32_t n) {
// this function will call LU underneath it and do the iterative updates needed
	std::cout<<"Inside GN baby boy";
  // // Set lower triangular diagonal to 1
	// for (int k = 0; k < n; ++k) {
	// 	L->ptr[k*n+k] = 1;
	// }
	// for (int k = 0; k < n; ++k) {
  //       // U matrix (upper triangular)
  //       for (int j = k; j < n; ++j) {
  //           U->ptr[k*n+j] = a.ptr[k*n+j];
  //           for (int i = 0; i < k; ++i) {
  //               U->ptr[k*n+j] -= L->ptr[k*n+i] * U->ptr[i*n+j];
  //           }
  //       }

  //       // L matrix (lower triangular)
  //       for (int i = k + 1; i < n; ++i) {
  //           L->ptr[i*n+k] = a.ptr[i*n+k];
  //           for (int j = 0; j < k; ++j) {
  //               L->ptr[i*n+k] -= L->ptr[i*n+j] * U->ptr[j*n+k];
  //           }
  //           L->ptr[i*n+k] /= U->ptr[k*n+k];
  //       }
  //   }
}

void ForwardBackward(const AlignedArray& L, const AlignedArray& U, const AlignedArray& y,
		                 AlignedArray* out, uint32_t n) {
	for (int i = 0; i < n; ++i) {
        out->ptr[i] = y.ptr[i];
        for (int j = 0; j < i; ++j) {
            out->ptr[i] -= L.ptr[i*n+j] * out->ptr[j];
        }
        out->ptr[i] /= L.ptr[i*n+i];
    }

	for (int i = n - 1; i >= 0; --i) {
        out->ptr[i] = out->ptr[i];
        for (int j = i + 1; j < n; ++j) {
            out->ptr[i] -= U.ptr[i*n+j] * out->ptr[j];
        }
        out->ptr[i] /= U.ptr[i*n+i];
    }

}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
    for (int i = 0; i < TILE; i++) {
    for (int k = 0; k < TILE; k++) {
    for(int j =0; j<TILE;j++){
      out[i * TILE + k] += a[i * TILE + j] * b[j * TILE + k];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
   for (int i = 0; i < (m / TILE); i++) {
    for (int j = 0; j < (p / TILE); j++) {
      
      //out->ptr[i*p/TILE  + j] = 0.0;
      //std::cout<<i<<j<<std::endl;
      AlignedArray temp_output = AlignedArray(TILE*TILE);
      //initialize to 0
      for (int k=0;k<TILE*TILE;k++)temp_output.ptr[k] = 0;
    

      AlignedArray lhs_mul = AlignedArray(TILE*TILE);
      AlignedArray rhs_mul = AlignedArray(TILE*TILE);
      for (int k = 0; k < (n / TILE); ++k) 
      {
      //retrieve vals
      for(int sm=0; sm<TILE*TILE;sm++)
      {
      lhs_mul.ptr[sm] = a.ptr[i*(n / TILE)*TILE*TILE + k*TILE*TILE+sm];
      rhs_mul.ptr[sm] = b.ptr[k*(p / TILE)*TILE*TILE + j*TILE*TILE+sm];
      //std::cout<<lhs_mul.ptr[sm];
       //std::cout<<a.ptr[i*(n / TILE)*TILE*TILE + k*TILE*TILE+sm];
      }
      //cal it
      AlignedDot(lhs_mul.ptr, rhs_mul.ptr, temp_output.ptr);
      
      }
      for (int sm=0; sm<TILE*TILE; sm++)out->ptr[i*p/TILE*TILE*TILE+j*TILE*TILE+sm] = temp_output.ptr[sm];
  
    }
  }
  }

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  // assert(a.size % reduce_size == 0);
  // assert(a.size / reduce_size == out->size);
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = std::numeric_limits<scalar_t>::min();
  }
  for (int i = 0; i < a.size; i++) {
     size_t reduced_index = i / reduce_size;
     size_t within_block_index = i % reduce_size;


    if (within_block_index == 0) {
      out->ptr[reduced_index] = a.ptr[i];
    } else if (a.ptr[i] > out->ptr[reduced_index]) {
      out->ptr[reduced_index] = a.ptr[i];
    }
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
    // Initialize the output array to 0
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = 0.0;
  }

  // Perform the reduction
  for (size_t i = 0; i < a.size; i++) {
    size_t reduced_index = i / reduce_size;
    out->ptr[reduced_index] += a.ptr[i];
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("LU", LUDecomposition);
  m.def("forward_backward", ForwardBackward);
  m.def("GN", GaussNewton);
}
