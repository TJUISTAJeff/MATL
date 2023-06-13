( tensor : an n dimensional torch tensor ) ( mean : the mean of the normal distribution ) ( std : the standard deviation of the normal distribution )
( alpha : the α\alphaα value for the elu formulation  default  1 0 ) ( inplace : can optionally do the operation in place  default  false )
( in channels  int  : number of channel in the input image ) ( out channels  int  : number of channel produce by the convolution ) ( kernel size  int or tuple  : size of the convolve kernel ) ( stride  int or tuple  optional  : stride of the convolution  default  1 ) ( padding  int or tuple  optional  : dilation    kernel size   1    pad zero pad will be add to both side of each dimension in the input  default  0 ) ( output padding  int or tuple  optional  : additional size add to one side of each dimension in the output shape  default  0 ) ( groups  int  optional  : number of block connections from input channel to output channel  default  1 ) ( bias  bool  optional  : if true  add a learnable bias to the output  default  true ) ( dilation  int or tuple  optional  : space between kernel elements  default  1 )
( in features : size of each input sample ) ( out features : size of each output sample ) ( bias : if set to false  the layer will not learn an additive bias  default  true )
( tensor : an n dimensional torch tensor ) ( a : the lower bind of the uniform distribution ) ( b : the upper bind of the uniform distribution )

( input : input tensor  minibatch in channel ih iw  \text minibatch    \text in\ channel    ih   iw  minibatch in channel ih iw   minibatch dim optional  ) ( kernel size : size of the pool region  can be a single number or a tuple  kh  kw  ) ( stride : stride of the pool operation  can be a single number or a tuple  sh  sw   default  kernel size ) ( padding : implicit negative infinity pad to be add on both side  must be >  0 and <  kernel size / 2  ) ( dilation : the stride between elements within a slide window  must be > 0  ) ( ceil mode : if true  will use ceil instead of floor to compute the output shape  this ensure that every element in the input tensor be cover by a slide window  ) ( return indices : if true  will return the argmax along with the max value  useful for torch nn functional max unpool2d later )
( p : probability of an element to be zero  default  0 5 ) ( training : apply dropout if be true  default  true ) ( inplace : if set to true  will do this operation in place  default  false )
( params  iterable  : iterable of parameters to optimize or dicts define parameter group ) ( lr  float  optional  : learn rate  default  1e 2  ) ( momentum  float  optional  : momentum factor  default  0  ) ( alpha  float  optional  : smooth constant  default  0 99  ) ( eps  float  optional  : term add to the denominator to improve numerical stability  default  1e 8  ) ( centered  bool  optional  : if true  compute the center rmsprop  the gradient be normalize by an estimation of its variance ) ( weight decay  float  optional  : weight decay  l2 penalty   default  0  )
( params  iterable  : iterable of parameters to optimize or dicts define parameter group ) ( lr  float  optional  : learn rate  default  1e 2  ) ( lr decay  float  optional  : learn rate decay  default  0  ) ( weight decay  float  optional  : weight decay  l2 penalty   default  0  ) ( eps  float  optional  : term add to the denominator to improve numerical stability  default  1e 10  )
( f : a file like object  have to implement read    readline    tell    and seek     or a string or os pathlike object contain a file name ) ( map location : a function  torch device  string or a dict specify how to remap storage locations ) ( pickle module : module use for unpickling metadata and object  have to match the pickle module use to serialize file  ) ( pickle load args :  python 3 only  optional keyword arguments pass over to pickle module load   and pickle module unpickler    e g   errors      )
( downscale factor  int  : factor to decrease spatial resolution by )
( padding  int  tuple  : the size of the pad  if be int  use the same pad in all boundaries  if a 4 tuple  use  pad left\text padding\ leave pad leave  pad right\text padding\ right pad right  pad top\text padding\ top pad top  pad bottom\text padding\ bottom pad bottom  )
( kernel size : the size of the window ) ( stride : the stride of the window  default value be kernel size ) ( padding : implicit zero pad to be add on all three side ) ( ceil mode : when true  will use ceil instead of floor to compute the output shape ) ( count include pad : when true  will include the zero pad in the average calculation ) ( divisor override : if specify  it will be use as divisor  otherwise kernel size will be use )
( inplace : if set to true  will do this operation in place  default  false )
( in channels  int  : number of channel in the input image ) ( out channels  int  : number of channel produce by the convolution ) ( kernel size  int or tuple  : size of the convolve kernel ) ( stride  int or tuple  optional  : stride of the convolution  default  1 ) ( padding  int  tuple or str  optional  : pad add to all six side of the input  default  0 ) ( padding mode  string  optional  : 'zeros'  'reflect'  'replicate' or 'circular'  default  'zeros' ) ( dilation  int or tuple  optional  : space between kernel elements  default  1 ) ( groups  int  optional  : number of block connections from input channel to output channel  default  1 ) ( bias  bool  optional  : if true  add a learnable bias to the output  default  true )

( input  Tensor  : the input tensor ) ( sorted  bool  : whether to sort the unique elements in ascend order before return as output  ) ( return inverse  bool  : whether to also return the indices for where elements in the original input end up in the return unique list  ) ( return counts  bool  : whether to also return the count for each unique element  ) ( dim  int  : the dimension to apply unique  if none  the unique of the flatten input be return  default  none )
( negative slope : control the angle of the negative slope  default  1e 2 ) ( inplace : can optionally do the operation in place  default  false )
( a  Tensor  : leave tensor to contract ) ( b  Tensor  : right tensor to contract ) ( dims  int or Tuple List int   List int   or List List int   containing two lists or Tensor  : number of dimension to contract or explicit list of dimension for a and b respectively )
( input size : the number of expect feature in the input x ) ( hidden size : the number of feature in the hide state h ) ( num layers : number of recurrent layer  e g   set num layer 2 would mean stack two rnns together to form a stack rnn  with the second rnn take in output of the first rnn and compute the final result  default  1 ) ( nonlinearity : the non linearity to use  can be either 'tanh' or 'relu'  default  'tanh' ) ( bias : if false  then the layer do not use bias weight b ih and b hh  default  true ) ( batch first : if true  then the input and output tensors be provide as  batch  seq  feature  instead of  seq  batch  feature   note that this do not apply to hide or cell state  see the inputs/outputs section below for detail   default  false ) ( dropout : if non zero  introduce a dropout layer on the output of each rnn layer except the last layer  with dropout probability equal to dropout  default  0 ) ( bidirectional : if true  become a bidirectional rnn  default  false )
( outputs  sequence of Tensor  : output of the differentiate function  ) ( inputs  sequence of Tensor  : input w r t  which the gradient will be return  and not accumulate into  grad   ) ( grad outputs  sequence of Tensor  : the “vector” in the vector jacobian product  usually gradients w r t  each output  none value can be specify for scalar tensors or ones that don’t require grad  if a none value would be acceptable for all grad tensors  then this argument be optional  default  none  ) ( retain graph  bool  optional  : if false  the graph use to compute the grad will be free  note that in nearly all case set this option to true be not need and often can be work around in a much more efficient way  default to the value of create graph  ) ( create graph  bool  optional  : if true  graph of the derivative will be construct  allow to compute higher order derivative products  default  false  ) ( allow unused  bool  optional  : if false  specify input that be not use when compute output  and therefore their grad be always zero  be an error  default to false  ) ( is grads batched  bool  optional  : if true  the first dimension of each tensor in grad output will be interpret as the batch dimension  instead of compute a single vector jacobian product  we compute a batch of vector jacobian products for each “vector” in the batch  we use the vmap prototype feature as the backend to vectorize call to the autograd engine so that this computation can be perform in a single call  this should lead to performance improvements when compare to manually loop and perform backward multiple time  note that due to this feature be experimental  there may be performance cliffs  please use torch  c  debug only display vmap fallback warn true  to show any performance warn and file an issue on github if warn exist for your use case  default to false  )
( kernel size : the size of the window ) ( stride : the stride of the window  default value be kernel size ) ( padding : implicit zero pad to be add on both side ) ( ceil mode : when true  will use ceil instead of floor to compute the output shape ) ( count include pad : when true  will include the zero pad in the average calculation ) ( divisor override : if specify  it will be use as divisor  otherwise size of the pool region will be use  )

( params  iterable  : iterable of parameters to optimize or dicts define parameter group ) ( lr  float  : learn rate ) ( momentum  float  optional  : momentum factor  default  0  ) ( weight decay  float  optional  : weight decay  l2 penalty   default  0  ) ( dampening  float  optional  : dampen for momentum  default  0  ) ( nesterov  bool  optional  : enable nesterov momentum  default  false  ) ( maximize  bool  optional  : maximize the params base on the objective  instead of minimize  default  false  )
( p : probability of an element to be zero  default  0 5 ) ( inplace : if set to true  will do this operation in place  default  false )
( kernel size : the size of the window ) ( stride : the stride of the window  default value be kernel size ) ( padding : implicit zero pad to be add on both side ) ( ceil mode : when true  will use ceil instead of floor to compute the output shape ) ( count include pad : when true  will include the zero pad in the average calculation )
( probs  Number  Tensor  : the probability of sample 1 ) ( logits  Number  Tensor  : the log odds of sample 1 )
( loc  float or Tensor  : mean of the distribution  often refer to as mu  ) ( scale  float or Tensor  : standard deviation of the distribution  often refer to as sigma  )
( inplace : can optionally do the operation in place  default  false )
( tensor : an n dimensional torch tensor ) ( gain : an optional scale factor )

( tensor : an n dimensional torch tensor ) ( a : the negative slope of the rectifier use after this layer  only use with 'leaky relu'  ) ( mode : either 'fan in'  default  or 'fan out'  choose 'fan in' preserve the magnitude of the variance of the weight in the forward pass  choose 'fan out' preserve the magnitudes in the backwards pass  ) ( nonlinearity : the non linear function  nn functional name   recommend to use only with 'relu' or 'leaky relu'  default   )