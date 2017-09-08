import tensorflow as tf

def conv2D(x,shape,name,stride=[1,2,2,1],padding='SAME',reuse=None):
	"""
	Computes 2D convolution

	Inputs
	------------------------
	x : input tensor
	shape : kernel shape [filter height, filter width, num featuremaps]
	stride : stride [batch size,filter height, filter width, in channels]
	padding : pad type
	reuse : bool wether to reuse the variables
	"""
	#get final shape
	input_channels = x.shape[-1]
	shape.append(shape[-1])
	shape[2]=input_channels

	#compute convolution
	with tf.variable_scope('conv2d_'+name,reuse=reuse):
		kernel = tf.get_variable(name='kernel',
									shape=shape,
									initializer=tf.truncated_normal_initializer(mean=0.,stddev=.01))
		y = tf.nn.conv2d(input=x,
							filter=kernel,
							strides=stride,
							padding=padding)
	return y


def dense(x,num_nodes,activation='relu',name='',bias_init_val=0,reuse=None):
    """
    Dense fully connected layer
    y = activation(xW+b)
    
    Inputs
    ------------------
    x : input tensor
    
    num_nodes : number of nodes in the layer
    
    acitvation (optional: activation function to use 
         one of ['relu','sigmoid',None]
         default is 'relu'
    
    name_suffix (optional) : the suffix to append to variable names
    
    bias_init_val (optional) : the initial value of the bias
    
    Outputs
    ---------------------
    y : the output tensor
    """
    input_shape = x.get_shape()
    with tf.variable_scope('dense',reuse=reuse):
    	W=tf.get_variable('W'+name,initializer=tf.random_normal(stddev=.01,shape=[int(input_shape[-1]),num_nodes]))
    	b=tf.get_variable('b'+name,initializer=tf.constant(bias_init_val,shape = [num_nodes],dtype=tf.float32))
    
    logits = tf.matmul(x,W)+b
    
    if activation == None:
        y = logits
    elif activation == 'relu':
        y = tf.nn.relu(logits)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(logits)
    else:
        raise ValueError("Enter a valid activation function")
    
    return y