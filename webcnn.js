/**
 * WebCNN - Browser-based Convolutional Neural Network
 * © Adam Smith 2017
 * MIT License
 */
const LAYER_TYPE_CONV =  'convLayer';
const LAYER_TYPE_MAX_POOL =  'maxPoolLayer';
const LAYER_TYPE_INPUT_IMAGE =  'inputImageLayer';
const LAYER_TYPE_INPUT =  'inputLayer';
const LAYER_TYPE_FULLY_CONNECTED =  'FCLayer';
const ACTIVATION_RELU =  'relu';
const ACTIVATION_TANH =  'tanh';
const ACTIVATION_LINEAR =  'linear';
const ACTIVATION_SOFTMAX=  'softmax';
const INIT_ZEROS =  'initZeros';
const INIT_XAVIER =  'xavier';
const INIT_GLOROT_UNIFORM =  'glorot_uniform';

const FORWARD_MODE_TRAINING =  'training';		// Normal training mode
const FORWARD_MODE_EVALUATE =  'evaluate';		// Normal evaluation mode where no values need to be calculated for backprop
const BACKWARD_MODE_TRAINING =  'training';
const BACKWARD_MODE_DREAMING =  'dreaming';

/*
	This class is the network wrapper which owns hyperparameters and
	provides the input/output API to the Layers of the network composed within it
 */
class WebCNN
{
	constructor()
	{
		this.layers = [];
		this.nextLayerIndex = 0;

		// Defaults to training mode
		this.forwardMode = FORWARD_MODE_TRAINING;
		this.backwardMode = BACKWARD_MODE_TRAINING;

		// Default hyperparameters
		this.learningRate = 0.01;
		this.momentum = 0.9;

		// L2 Regularization
		this.lambda =  0.0;

		// For auto-detection of solved MLP networks
		this.solutionEpoch = 0;
		this.trainingError = 0;

		// For benchmarking performance
		this.forwardTime = 0;
		this.backwardTime = 0;
	}

	// Public API getters and setters
	setMomentum( momentum ) { this.momentum = momentum; }
	getMomentum() { return this.momentum; }

	setLearningRate( rate ) { this.learningRate = rate; }
	getLearningRate() { return this.learningRate; }

	setLambda( lambda ) { this.lambda = lambda; }
	getLambda() { return this.lambda; }

	// Create a layer from a descriptor object and add it to the layer stack
	// intention: public API
	newLayer( layerDesc )
	{
		console.log( "Creating layer "+this.nextLayerIndex+": "+layerDesc.name );

		let newLayer;

		switch( layerDesc.type )
		{
			case LAYER_TYPE_INPUT_IMAGE:
			{
				newLayer = new InputImageLayer( layerDesc.name, layerDesc.width, layerDesc.height, layerDesc.depth );
				break;
			}

			case LAYER_TYPE_CONV:
			{
				newLayer = new ConvLayer(	layerDesc.name, layerDesc.units, layerDesc.kernelWidth, layerDesc.kernelHeight,
											layerDesc.strideX, layerDesc.strideY, layerDesc.padding );
				break;
			}

			case LAYER_TYPE_MAX_POOL:
			{
				newLayer = new MaxPoolLayer( layerDesc.name, layerDesc.poolWidth, layerDesc.poolHeight,
											 layerDesc.strideX, layerDesc.strideY );
				break;
			}

			case LAYER_TYPE_FULLY_CONNECTED:
			{
				newLayer = new FCLayer( layerDesc.name, layerDesc.units, layerDesc.activation );
				break;
			}
		}

		newLayer.network = this;

		if ( this.nextLayerIndex == 0 )
		{
			if ( newLayer.type != LAYER_TYPE_INPUT_IMAGE && newLayer.type != LAYER_TYPE_INPUT )
			{
				throw "First layer must be input type";
			}
		}
		else
		{
			// Link with previous layer
			const prevLayer = this.layers[ this.nextLayerIndex - 1 ];
			prevLayer.setOutputLayer(newLayer);
			newLayer.setInputLayer(prevLayer);
		}

		newLayer.layerIndex = this.nextLayerIndex;
		this.layers[ this.nextLayerIndex ] = newLayer;
		this.nextLayerIndex++;
	}

	// intention: public API
	initialize()
	{
		// Placeholder. Right now, the last-added layer is implicitly the output, but it's
		// my intention to formalize "finalizing" and initializing some parameters of the
		// network in this function.
	}

	// intention: public API
	trainCNNClassifier( imageDataArray, imageLabelsArray )
	{
		this.miniBatchSize = Math.floor( Math.min( imageDataArray.length, imageLabelsArray.length ) );

		this.batchLearningRate = this.learningRate / this.miniBatchSize;

		this.forwardMode = FORWARD_MODE_TRAINING;
		this.backwardMode = BACKWARD_MODE_TRAINING;
		this.trainingError = 0;

		this._cnnSetTrainingTargets( imageLabelsArray );

		let t0 = Date.now();
		this._cnnForward( imageDataArray );
		let t1 = Date.now();
		this._cnnBackward();
		this._endCNNMiniBatch();

		this.trainingError /= this.miniBatchSize;
		this.forwardTime = ( t1 - t0 ) / this.miniBatchSize;
		this.backwardTime = ( Date.now() - t1) / this.miniBatchSize;
	}

	// intention: private
	_cnnSetTrainingTargets( imageLabelsArray )
	{
		this.targetValues = new Array( this.miniBatchSize );
		for ( var example = 0; example < this.miniBatchSize; ++example )
		{
			this.targetValues[ example ] = new Array( this.layers[ this.layers.length - 1 ].units );
			for ( var unit = 0; unit < this.layers[ this.layers.length - 1 ].units; ++unit )
			{
				this.targetValues[ example ][ unit ] = ( unit == imageLabelsArray[ example ] ) ? 1 : 0;
			}
		}
	}

	// intention: private
	_cnnForward( imageDataArray )
	{
		this.layers[ 0 ].forward( imageDataArray );
		for ( let i = 1; i < this.layers.length; ++i )
		{
			this.layers[ i ].forward();
		}
	}

	// intention: private
	_cnnBackward()
	{
		for ( let i = this.layers.length - 1; i > 0; --i )
		{
			this.layers[ i ].backward();
		}
	}

	// intention: private
	_endCNNMiniBatch()
	{
		for ( let i = this.layers.length - 1; i > 0; --i )
		{
			this.layers[ i ].commitMiniBatch();
		}
	}

	// intention: public API
	classifyImages( imageDataArray )
	{
		let outputLayer = this.layers[ this.layers.length - 1 ];
		this.miniBatchSize = imageDataArray.length;
		this.batchLearningRate = this.learningRate;

		this.forwardMode = FORWARD_MODE_EVALUATE;

		this.layers[ 0 ].forward( imageDataArray );
		for ( var i = 1; i < this.layers.length; ++i )
		{
			this.layers[ i ].forward();
		}

		this.forwardMode = FORWARD_MODE_TRAINING;
		return outputLayer.output;
	}
}

/*
	Abstract network layer base class
 */
class Layer
{
	/*
	 name: String, the name of the layer used for debugging and logging
	 units: Number of neurons/nodes/units in a FC layer, or kernels in a conv layer
	 inputs: Number of incoming connections, required for determining variance in Xavier style weight initialization
	 */
	constructor( name, units )
	{
		this.name = name;
		this.units = units;

		// Defaults
		this.activation = ACTIVATION_LINEAR;
	}

	isLastLayer() { return this.nextLayer == undefined; }

	setInputLayer( inputLayer )
	{
		this.prevLayer = inputLayer;
		this.inputDimensions = inputLayer.outputDimensions;
	}

	setOutputLayer( outputLayer )
	{
		this.nextLayer = outputLayer;
	}

	commitMiniBatch()
	{
		// No-op in layers that don't override this (layers without weights and biases)
	}
}

/*
 	Input values layer, just some numbers. This layer option is for making small
 	non-convolutional networks for testing and debugging purposes.
 */
class InputLayer extends Layer
{
	constructor( name, units )
	{
		super( name, units, 0, units );
		this.type = LAYER_TYPE_INPUT;
		this.activation = ACTIVATION_NONE;
		this.outputDimensions = new Dimensions( 1, 1, units );
	}

	forward()
	{
		for ( var example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );
			for ( let unit = 0; unit < this.units; ++unit )
			{
				this.output[ example ].setValue( 0, 0, unit, this.network.inputValues[ example ][ unit ] );
			}
		}
	}
}

/*
 	Input image layer
 */
class InputImageLayer extends Layer
{
	constructor( layerName, imageWidth, imageHeight, imageDepth )
	{
		if ( imageDepth != 1 && imageDepth != 3 )
		{
			// Only 1 or 3 components per pixel are supported (i.e. greyscale or 3-component color)
			throw "Invalid input image depth, must be 1 (greyscale) or 3 (RGB)";
		}

		super( layerName, 1 );
		this.type = LAYER_TYPE_INPUT_IMAGE;
		this.outputDimensions = new Dimensions( imageWidth, imageHeight, imageDepth );
		this.output = [];
	}

	// This layer always just sets output values on forward propagation
	// imageDataArray must be an Array of ImageData objects
	forward( imageDataArray )
	{
		for ( var example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );
			this.output[ example ].setFromImageData( imageDataArray[ example ], this.outputDimensions.depth );
		}
	}

	// In normal classification/regression, back propagation does not alter the input image,
	// this function is only for deep dreaming style image reconstruction.
	backward()
	{
		// Reserved for future functionality
	}
}


/*
 	Fully-connected layer
 */
class FCLayer extends Layer
{
	constructor( name, units, activation )
	{
		super( name, units );
		this.type = LAYER_TYPE_FULLY_CONNECTED;
		this.biases = new Float32Array( units );
		this.outputDimensions = new Dimensions( 1, 1, units );
		this.activation = activation;
	}

	setInputLayer( inputLayer )
	{
		super.setInputLayer( inputLayer );
		this.weights = new Array( this.units );
		this.biasVelocities = new Float32Array( this.units );
		this.weightVelocities = new Array( this.units );

		for ( var unit = 0; unit < this.units; ++unit )
		{
			this.weights[ unit ] = new Array3D( this.inputDimensions, INIT_XAVIER );
			this.weightVelocities[ unit ] = new Array3D( this.inputDimensions, INIT_ZEROS );

		}
	}

	setOutputLayer( outputLayer )
	{
		super.setOutputLayer( outputLayer );
	}

	setWeightsAndBiases( weights, biases )
	{
		for ( var unit = 0; unit < this.units; ++unit )
		{
			this.weights[ unit ].setFromArray( weights[ unit ] );
		}
		this.biases = new Float32Array( biases );
	}

	forward()
	{
		this.input = this.prevLayer.output;
		this.output = new Array( this.network.miniBatchSize );
		const size = this.inputDimensions.getSize();

		for ( var example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );

			// Dot product of inputs with weights
			for ( var unit = 0; unit < this.units; ++unit )
			{
				for ( var i = 0; i < size; ++i )
				{
					this.output[ example ].values[ unit ] += this.weights[ unit ].values[ i ] * this.input[ example ].values[ i ];
				}
				this.output[ example ].values[ unit ] += this.biases[ unit ];
			}

			this.output[ example ].applyActivationFunction( this.activation );
		}
	}

	backward()
	{
		var example, unit;

		this.backpropError = new Array( this.network.miniBatchSize );

		// Softmax and linear (gain=1) error terms work out to the same value, tanh has an extra derivative term
		for ( example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.backpropError[ example ] = new Array3D( this.inputDimensions, INIT_ZEROS );

			for ( unit = 0; unit < this.units; ++unit )
			{
				var error;
				if (this.isLastLayer())
				{
					error = this.output[ example ].values[ unit ] - this.network.targetValues[ example ][ unit ];
				}
				else
				{
					error = this.nextLayer.backpropError[ example ].getValue( 0, 0, unit );
				}

				if ( this.activation == ACTIVATION_TANH )
				{
					error *= ( 1.0 - Math.pow( this.output[ example ].values[ unit ], 2 ) );
				}

				this.network.trainingError += Math.abs( error );

				// Velocity updates
				var update = -this.network.batchLearningRate * error;
				this.weightVelocities[ unit ].addScaledArray3D( update, this.input[ example ] );
				this.biasVelocities[ unit ] += update;

				this.backpropError[ example ].addScaledArray3D( error, this.weights[ unit ] );

			}
		}
	}

	// Commits accumulated changes to weights and biases and resets update arrays to zeroes for the next mini-batch
	commitMiniBatch()
	{
		const L2reg = 1.0 - ( this.network.batchLearningRate * this.network.lambda );

		for ( var unit = 0; unit < this.units; ++unit )
		{
			// L2 Regularization - subtracts weight decay term, via multiplication with (1.0 - term)
			// for weights only, not done to biases
			this.weights[ unit ].scaleThenAddArray3D( L2reg, this.weightVelocities[ unit ] );
			this.biases[ unit ] += this.biasVelocities[ unit ];

			// Apply momentum to velocities for next batch
			this.biasVelocities[ unit ] *= this.network.momentum;
			this.weightVelocities[ unit ].scaleBy( this.network.momentum );
		}
	}
}

/*
 Convolution layer
 */
class ConvLayer extends Layer
{
	constructor( name, units, kernelWidth, kernelHeight, kernelStrideX, kernelStrideY, usePadding )
	{
		super( name, units );

		this.type = LAYER_TYPE_CONV;
		this.activation = ACTIVATION_RELU;
		this.inputLayer = undefined;

		this.kernelWidth = kernelWidth;
		this.kernelHeight = kernelHeight;
		this.kernelDepth = undefined;

		// TODO: Support zero-padding and strides > 1
		this.kernelStrideX = 1;
		this.kernelStrideY = 1;
		this.padX = 0;
		this.padY = 0;

		this.biases = new Float32Array( units );
		this.kernels = new Array( units );
		this.errorDeltas = undefined;
		this.backpropError = undefined;
	}

	setInputLayer( inputLayer )
	{
		super.setInputLayer( inputLayer );

		const outputWidth = Math.floor( ( this.inputDimensions.width + this.padX * 2 - this.kernelWidth) / this.kernelStrideX + 1 );
		const outputHeight = Math.floor( ( this.inputDimensions.height + this.padY * 2 - this.kernelHeight) / this.kernelStrideY + 1 );

		// Output dimensions is the size of each unit's activation map, times the number of units
		this.outputDimensions = new Dimensions( outputWidth, outputHeight, this.units );

		this.kernelDepth = this.inputDimensions.depth;

		const kernelDimensions = new Dimensions( this.kernelWidth, this.kernelHeight, this.kernelDepth );

		this.biasVelocities = new Float32Array( this.units );
		this.kernelVelocities = new Array( this.units );

		for ( var unit = 0; unit < this.units; ++unit )
		{
			this.kernels[ unit ] = new Array3D( kernelDimensions, INIT_XAVIER );
			this.kernelVelocities[ unit ] = new Array3D( kernelDimensions, INIT_ZEROS );
		}
	}

	setWeightsAndBiases( weights, biases )
	{
		console.log( weights, biases );
		for ( var unit = 0; unit < this.units; ++unit )
		{
			this.kernels[ unit ].setFromArray( weights[ unit ] );
		}
		this.biases = new Float32Array( biases );
	}

	forward()
	{
		var example, unit, oy, ox, ky, kx, kd, iy, ix;

		// Each unit has a kernel, and produces an output, which is an activation map (a floating-point image, basically)
		this.input = this.prevLayer.output;
		this.output = new Array( this.network.miniBatchSize );

		for ( example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );

			for ( unit = 0; unit < this.units; ++unit )
			{
				for ( oy = 0; oy < this.outputDimensions.height; ++oy )
				{
					for ( ox = 0; ox < this.outputDimensions.width; ++ox )
					{
						// For each pixel in the output activation map, do cross-correlation of input and kernel
						for ( ky = 0; ky < this.kernelHeight; ++ky )
						{
							iy = oy + ky;
							for ( kx = 0; kx < this.kernelWidth; ++kx )
							{
								ix = ox + kx;
								for ( kd = 0; kd < this.kernelDepth; ++kd )
								{
									this.output[ example ].addToValue( ox, oy, unit, this.kernels[ unit ].getValue( kx, ky, kd ) * this.input[ example ].getValue( ix, iy, kd ) );
								}
							}
						}
						this.output[ example ].addToValue( ox, oy, unit, this.biases[ unit ] );
					}
				}
			}

			this.output[ example ].applyActivationFunction( this.activation );
		}
	}

	backward()
	{
		var nextLayer = this.nextLayer;
		var example, unit, oy, ox, kd, ky, kx, iy, ix, scaledError;
		var errorDelta;

		// Convolution of errorDeltas with filter weights to produce error term that propagates back to previous layer
		// to be then multiplied by that layer's activation derivative to become that layer's errorDeltas
		this.backpropError = new Array( this.network.miniBatchSize );

		for ( example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.backpropError[ example ] = new Array3D( this.inputDimensions, INIT_ZEROS );

		}

		// Calculating filter weight updates requires cross-correlation of this layer's inputs (prev layer's output) with
		// this layer's errorDelta values
		//
		// Calculating the backpropagating error that will be passed to the previous layer requires a convolution of this
		// layer's errorDeltas matrix with this layer's filter kernel values.
		//
		// Because both of these convolution/correlation operations involve the errorDeltas matrix, and exactly the same
		// number of multiplications, they are combined below into a single set of nested loops which perform both operations
		// concurrently.
		for ( example = 0; example < this.network.miniBatchSize; ++example )
		{
			for ( unit = 0; unit < this.units; ++unit )
			{
				for ( oy = 0; oy < this.outputDimensions.height; ++oy )
				{
					for ( ox = 0; ox < this.outputDimensions.width; ++ox )
					{
						// Error coming back from the next layer to this unit. When multiplied by this unit's activation function
						// gradient ( sigmoid prime of the unit input sum ), the resulting value is the error term normally
						// notated by lowercase delta in the literature.
						errorDelta = nextLayer.backpropError[ example ].getValue( ox, oy, unit ) * ( ( this.output[ example ].getValue( ox, oy, unit ) > 0 ) ? 1 : 0 );
						scaledError = -this.network.batchLearningRate * errorDelta;
						this.biasVelocities[ unit ] += scaledError;

						for ( kd = 0; kd < this.kernelDepth; ++kd )
						{
							for ( ky = 0; ky < this.kernelHeight; ++ky )
							{
								iy = ky + oy;
								for ( kx = 0; kx < this.kernelWidth; ++kx )
								{
									ix = kx + ox;

									// The kernel weights update cross-correlation of: input (X) errorDeltas
									this.kernelVelocities[ unit ].addToValue( kx, ky, kd, scaledError * this.input[ example ].getValue( ix, iy, kd ) );

									// The backprop error convolution of: weights (*) errorDeltas
									this.backpropError[ example ].addToValue( ix, iy, kd, errorDelta * this.kernels[ unit ].getValue( kx, ky, kd ) );
								}
							}
						}
					}
				}
			}
		}
	}

	// This function commits accumulated weight changes (saved in the velocities arrays) to the
	// kernel weights.
	commitMiniBatch()
	{
		const L2reg = 1.0 - ( this.network.batchLearningRate * this.network.lambda );

		for ( var unit = 0; unit < this.units; ++unit )
		{
			this.kernels[ unit ].scaleThenAddArray3D( L2reg, this.kernelVelocities[ unit ] );
			this.biases[ unit ] += this.biasVelocities[ unit ];

			this.kernelVelocities[ unit ].scaleBy( this.network.momentum );
			this.biasVelocities[ unit ] *= this.network.momentum;
		}
	}
}

/*
	Max Pool Layer 2x2
 	Note that this implemenation supports only 2x2 pooling, as an optimization for being
 	able to store the activated cell indices in 8-bit bitfields.
 */
class MaxPoolLayer extends Layer
{
	constructor( name, poolWidth, poolHeight, strideX, strideY )
	{
		super( name, 0 );
		this.type = LAYER_TYPE_MAX_POOL;
		this.poolWidth = poolWidth;
		this.poolHeight = poolHeight;
		this.strideX = strideX;
		this.strideY = strideY;
		this.poolSize = poolWidth * poolHeight;
	}

	setInputLayer( inputLayer )
	{
		super.setInputLayer( inputLayer );

		this.outputWidth = Math.floor( ( this.inputDimensions.width - this.poolWidth) / this.strideX + 1 );
		this.outputHeight = Math.floor( ( this.inputDimensions.height - this.poolHeight) / this.strideY + 1 );
		this.outputDepth = this.inputDimensions.depth;

		// The output is the cube of max values
		this.outputDimensions = new Dimensions( this.outputWidth, this.outputHeight, this.outputDepth );
	}

	forward()
	{
		var d, y, x, sy, sx, iy, ix, max;
		var bitfield, bitCount;

		this.input = this.prevLayer.output;
		this.output = new Array( this.network.miniBatchSize );
		this.poolMaxActivationIndices = new Array( this.network.miniBatchSize );

		// During forward propagation, I save the indices of all the pixels in the pool that attained the maximum
		// value. Note that this can be as few as 1 or as many as all of the pixels in the poolWidth x poolHeight cell.
		// This differs from most max pool implementations, which typically randomly selected one off the maxed-out inputs.
		// That just doesn't sit well with me as a mathematician, so I've implemented max pool to evenly distribute the
		// backpropagated error across all maxed-out inputs. Because of this, poolMaxActivationIndices is a bitfield per pool.
		const flagBits = this.poolSize;
		const countBits = Math.ceil( Math.log( flagBits ) / Math.log( 2 ) );
		const totalBits = flagBits + countBits;
		let intBits = nextHighestPowerOfTwo( totalBits );

		if ( intBits < 8 ) intBits = 8;
		if ( intBits > 32 )
		{
			throw "Pool dimensions are too large, max supported size is 5x5";
		}

		for ( var example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.poolMaxActivationIndices[ example ] = new UIntCube( this.outputDimensions, intBits );

			this.output[ example ] = new Array3D( this.outputDimensions, INIT_ZEROS );

			for ( d = 0; d < this.outputDepth; ++d )
			{
				for ( y = 0; y < this.outputHeight; ++y )
				{
					for ( x = 0; x < this.outputWidth; ++x )
					{
						max = 0;
						for ( iy = 0; iy < this.poolHeight; ++iy )
						{
							sy = y * this.strideY + iy;
							for ( ix = 0; ix < this.poolWidth; ++ix )
							{
								sx = x * this.strideX + ix;
								max = Math.max( max, this.input[ example ].getValue( sx, sy, d ) );
							}
						}

						this.output[ example ].setValue( x, y, d, max );

						// During training, we need to save which pixels were at max value
						if ( this.network.forwardMode == FORWARD_MODE_TRAINING )
						{
							bitfield = 0;
							bitCount = -1; // Count is stored as (actual count - 1);

							if ( max > 0 )
							{
								for ( iy = 0; iy < this.poolHeight; ++iy )
								{
									sy = y * this.strideY + iy;
									for ( ix = 0; ix < this.poolWidth; ++ix )
									{
										sx = x * this.strideX + ix;
										if ( max <= this.input[ example ].getValue( sx, sy, d ) )
										{
											// Set a bit in the lower 4 bits indicating that this input was at max value
											bitfield |= ( 1 << ( iy * this.poolWidth + ix ) );
											bitCount++;
										}
									}
								}
							}
							// Use the high bits of the bitfield to store how many inputs were at max,
							// (1 to poolSize) stored as (0 to poolSize-1) to require fewer bits, since it's
							// not possible for zero pixels to be active. I save this value to avoid having to
							// determine it from counting the set bits again (i.e. computing the Hamming Weight).
							bitfield |= ( bitCount << this.poolSize );
							this.poolMaxActivationIndices[ example ].setValue( x, y, d, bitfield );
						}
					}
				}
			}
		}
	}

	backward()
	{
		const nextLayer = this.nextLayer;

		this.backpropError = new Array( this.network.miniBatchSize );

		var example, x, y, d, ix, iy, id, sx, sy, bit;
		var bitfield, numActivatedInputs;

		for ( example = 0; example < this.network.miniBatchSize; ++example )
		{
			this.backpropError[ example ] = new Array3D( this.inputDimensions, INIT_ZEROS );

			for ( ix = 0; ix < this.inputDimensions.width; ++ix )
			{
				for ( iy = 0; iy < this.inputDimensions.height; ++iy )
				{
					for ( id = 0; id < this.inputDimensions.depth; ++id )
					{
						// First figure out which pixel in this layer's output the input pixel mapped to
						x = Math.floor( ix / this.poolWidth );
						y = Math.floor( iy / this.poolHeight );

						// Pull the pooled pixel's activation bitfield
						bitfield = this.poolMaxActivationIndices[ example ].getValue( x, y, id );
						numActivatedInputs = ( bitfield >> this.poolSize ) + 1; // convert from [0 to poolSize-1] back to [1 to poolSize]

						// Now get the indices within the chunk of source image that maps to this single pooled pixel
						sx = ix % this.poolWidth;
						sy = iy % this.poolHeight;

						bit = 1 << ( sy * this.poolWidth + sx );
						if ( ( bitfield & bit ) > 0 )
						{
							// This input was at max, pass back the portion of the error delta (divided among activated inputs if there are more than one)
							this.backpropError[ example ].setValue( ix, iy, id, nextLayer.backpropError[ example ].getValue( x, y, id ) / numActivatedInputs );
						}
						else
						{
							this.backpropError[ example ].setValue( ix, iy, id, 0);
						}
					}
				}
			}
		}
	}
}

/*
	Array3D is a wrapper around a Float32Array that is conceptually (externally) a
	3-dimensional array of numbers, but is currently stored as a 1-dimensional array internally
	for performance reasons and compatibility with future porting to a WebGL implementation
 */
class Array3D
{
	constructor( dimensions, initType, unit )
	{
		var i;
		this.dimensions = dimensions;
		this.size = dimensions.getSize();
		this.values = new Float32Array( this.size );
		const stdDev = 1.0 / Math.sqrt( this.dimensions.width * this.dimensions.height * this.dimensions.depth );

		if ( initType == INIT_XAVIER )
		{
			// 1/inputs Xavier initialization based on He, Rang, Zhen and Sun recommendation for ReLU activations
			// number of inputs per activation is just the number of weights
			const rng = new GaussianRNG();
			for ( i = 0; i < this.size; ++i )
			{
				this.values[ i ] = stdDev * rng.getNextRandom();
			}
		}
		else if ( initType == INIT_GLOROT_UNIFORM )
		{
			for ( i = 0; i < this.size; ++i )
			{
				this.values[ i ] = stdDev * ( 2.0 * Math.random() - 1.0);
			}
		}
	}

	zeroFill()
	{
		this.values.fill(0);
	}

	scaleThenAddArray3D( scaleFactor, additionA3D )
	{
		for ( var i = 0; i < this.size; ++i )
		{
			this.values[ i ] *= scaleFactor;
			this.values[ i ] += additionA3D.values[ i ];
		}
	}

	addScaledArray3D( scaleFactor, additionA3D )
	{
		for ( var i = 0; i < this.size; ++i )
		{
			this.values[ i ] += scaleFactor * additionA3D.values[ i ];
		}
	}

	scaleBy( scaleFactor )
	{
		for ( var i = 0; i < this.size; ++i )
		{
			this.values[ i ] *= scaleFactor;
		}
	}

	applyActivationFunction( activationType )
	{
		let i;

		switch( activationType )
		{
			case ACTIVATION_RELU:
			{
				for ( i = 0; i < this.size; ++i )
				{
					this.values[ i ] = Math.max( 0, this.values[ i ] );
				}

				break;
			}

			case ACTIVATION_TANH:
			{
				for ( i = 0; i < this.size; ++i )
				{
					this.values[ i ] = Math.tanh( this.values[ i ] );
				}

				break;
			}

			case ACTIVATION_SOFTMAX:
			{
				var max = 0;
				var sumExp = 0;

				for ( i = 0; i < this.size; ++i )
				{
					max = Math.max( max, this.values[ i ] );
				}

				for ( i = 0; i < this.size; ++i )
				{
					// Move output values to have max of 0, so that summing exponentials doesn't overflow
					this.values[ i ] =  Math.exp( this.values[ i ] - max );
					sumExp += this.values[ i ];
				}

				for ( i = 0; i < this.size; ++i )
				{
					this.values[ i ] /= sumExp;
				}
				break;
			}

			// ACTIVATION_LINEAR is a no-op
		}
	}

	getValues()
	{
		return this.values;
	}

	getValuesAsArray()
	{
		return Array.from( this.values );
	}

	getSubArray( begin, end )
	{
		return this.values.subarray( begin, end );
	}

	setFromArray( array )
	{
		this.values = new Float32Array( array );
	}

	// scalar value multiplies by each pixel value, which start out as range 0-255. Can be used to control floating point range of input values.
	// depth should be 1 or 3, for greyscale or color image respectively. Alpha channel information is discarded.
	setFromImageData( imageData, depth )
	{
		const scaleFactor = 1.0 / 255.0;

		if ( depth != 1 && depth != 3 )
		{
			throw "Set Array3D from ImageData with unsupported depth";
		}

		if ( imageData.width != this.dimensions.width || imageData.height != this.dimensions.height || this.dimensions.depth != depth )
		{
			throw "Set Array3D from ImageData width, height or depth mismatch";
		}

		const channelBytes = imageData.width * imageData.height;
		for ( var z = 0, z_offset = 0; z < depth; ++z, z_offset += channelBytes )
		{
			for ( var y = 0, y_offset = 0; y < imageData.height; ++y, y_offset += imageData.width )
			{
				for ( var x = 0; x < imageData.width; ++x )
				{
					// ImageData.data arrays are always RGBA, so if we're treating as greyscale we still need to jump 4 bytes per
					// pixel, but only the red channel value is used (it's assumed that R==G==B redundantly)
					var pixelIndex = 4 * ( y_offset + x );
					var arrayIndex = z_offset + y_offset + x;

					// note that z is added to pixel index, because each pixel consists of 4 bytes, and z is the offset to the color channel
					this.values[ arrayIndex ] = imageData.data[ pixelIndex + z ] * scaleFactor;
				}
			}
		}
	}

	setValue( x, y, z, value )
	{
		const index = ( z * this.dimensions.height + y ) * this.dimensions.width + x;
		this.values[ index ] = value;
	}

	addToValue( x, y, z, value )
	{
		const index = ( z * this.dimensions.height + y ) * this.dimensions.width + x;
		this.values[ index ] += value;
	}

	getValue( x, y, z )
	{
		const index = ( z * this.dimensions.height + y ) * this.dimensions.width + x;
		return ( this.values[ index ] );
	}
}

/*
 Int8Cube is a wrapper around a Int8Array that is conceptually (externally) a
 3-dimensional dimensions of numbers, but is currently stored as a 1-dimensional array internally
 for performance reasons and compatibility with porting this to a WebGL implementation
 */
class UIntCube
{
	constructor( dimensions, bits )
	{
		this.dimensions = dimensions;
		const size = dimensions.getSize();

		switch (bits)
		{
			case 8: this.values = new Uint8Array( size ); break;
			case 16: this.values = new Uint16Array( size ); break;
			case 32: this.values = new Uint32Array( size ); break;
		}
	}

	setValue( x, y, z, value )
	{
		const index = ( z * this.dimensions.height + y ) * this.dimensions.width + x;
		this.values[ index ] = value;
	}

	getValue( x, y, z )
	{
		const index = ( z * this.dimensions.height + y ) * this.dimensions.width + x;
		return ( this.values[ index ] );
	}
}

/*
 A simple class for storing and comparing dimensions of input volumes, output volumes, kernels, etc.
 */
class Dimensions
{
	constructor( width, height, depth )
	{
		// Flooring all values to guarantee integer dimensions.
		this.width = Math.floor( width );
		this.height = Math.floor( height );
		this.depth = Math.floor( depth );
	}

	getSize()
	{
		return this.width * this.height * this.depth;
	}

	// For debugging sanity checks
	static equal( dim1, dim2 )
	{
		return ( dim1.width == dim2.width && dim1.height == dim2.height && dim1.depth == dim2.depth );
	}
}

// Returns the power of 2 equal to or the next one higher than the argument n
function nextHighestPowerOfTwo( n )
{
	return Math.pow( 2, Math.ceil( Math.log( n ) / Math.log( 2 ) ) );
}