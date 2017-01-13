let cnn;
window.onload = Init;

const totalTrainingImages = 60000;
let miniBatchSize = 20;
const testAccuracySampleSize = 100;

let iterations = 0;
let examplesSeen = 0;
let running = false;
let paused = false;
let testing = false;
let timeoutID;

let ctx_error;
let mostRecentTrainingError = 0;
let prevError;
let epoch = 0;
let prevAccuracy;
let plotX = 0;
let lastAccPlotX;

let trainingSetImages = new Array(6);
let trainingSetCTX = new Array(6);
let trainingSetImagesCounter = 0;
let trainingSetImagesLoaded = false;


let testSetImage;
let testSetImageLoaded = false;
let testSetCTX;

let exampleTimer = 0;
let forwardTime = 0;
let backwardTime = 0;

let epochsPerPixel;


function Init()
{
	let canvas = document.getElementById("weightsCanvas");
	ctx_error = canvas.getContext("2d");
	ctx_error.strokeStyle = "red";
	epochsPerPixel = Math.ceil( 60000 / ( miniBatchSize * ctx_error.canvas.width ) );

	// Load training data
	for ( let i = 0; i < 6; ++i )
	{
		trainingSetImages[i] = new Image();
		trainingSetImages[i].src = '../images/mnist_training_'+i+'.png';
		trainingSetImages[i].onload = OnTrainingSetImageLoaded;
	}

	testSetImage = new Image();
	testSetImage.src = '../images/mnist_test.png';
	testSetImage.onload = OnTestSetImageLoaded;

	document.getElementById('networkFile').addEventListener('change', networkFileChoosen, false);

	createDefaultNetwork();
}

function resetGraphForContinuedTraining()
{
	ctx_error.fillStyle = "white";
	ctx_error.fillRect(0,0, ctx_error.canvas.width, ctx_error.canvas.height);
	plotX = -1;
}

function resetGraphForNewNetwork()
{
	ctx_error.fillStyle = "white";
	ctx_error.fillRect(0,0, ctx_error.canvas.width, ctx_error.canvas.height);

	mostRecentTrainingError = 0;
	prevError = undefined;
	prevAccuracy = undefined;
	lastAccPlotX = undefined;
	epoch = 0;
	iterations = 0;
	examplesSeen = 0;
	plotX = -1;
}

function plotError( err )
{
	let currError = err;
	if (prevError != undefined && plotX > 0 )
	{
		ctx_error.strokeStyle = "red";
		ctx_error.beginPath();
		ctx_error.moveTo( (plotX - 1), ctx_error.canvas.height - prevError * ( ctx_error.canvas.height / 2 ) );
		ctx_error.lineTo( plotX, ctx_error.canvas.height - currError * ( ctx_error.canvas.height / 2 ) );
		ctx_error.stroke();
	}
	prevError = currError;
}

function plotAccuracy( accuracy )
{
	//console.log("plot accuracy="+accuracy+"  plotX="+plotX+" prevAccuracy="+prevAccuracy+" lastAccPlotX="+lastAccPlotX+" epoch="+epoch );
	if ( plotX > 0 )
	{
		ctx_error.strokeStyle = "blue";
		ctx_error.beginPath();
		ctx_error.moveTo( lastAccPlotX, ctx_error.canvas.height - prevAccuracy * ( ctx_error.canvas.height ) );
		ctx_error.lineTo( plotX, ctx_error.canvas.height - accuracy * ( ctx_error.canvas.height ) );
		ctx_error.stroke();
	}
	lastAccPlotX = plotX;
	prevAccuracy = accuracy;
}

function networkFileChoosen(evt)
{
	console.log("loaded");
	let files = evt.target.files; // FileList object

	let reader = new FileReader();
	reader.onload = OnNetworkJSONLoaded;
	reader.readAsText(files[0]);
}

function OnTrainingSetImageLoaded( e )
{
	trainingSetImagesCounter++;
	if ( trainingSetImagesCounter == trainingSetImages.length )
	{
		for ( var i = 0; i < trainingSetImages.length; ++i )
		{
			let canvas = document.createElement("canvas");
			canvas.width = 2800;
			canvas.height = 2800;
			trainingSetCTX[ i ] = canvas.getContext("2d");
			trainingSetCTX[ i ].drawImage( trainingSetImages[ i ], 0, 0);
		}
		trainingSetImagesLoaded = true;
		console.log("Training Images Loaded");
	}
}

function OnTestSetImageLoaded( e )
{
	let canvas = document.createElement("canvas");
	canvas.width = 2800;
	canvas.height = 2800;
	testSetCTX = canvas.getContext("2d");
	testSetCTX.drawImage( testSetImage, 0, 0);
	testSetImageLoaded = true;
	console.log("Test Images Loaded");
}

function buttonClick( n )
{
	let btn = document.getElementById("startButton");
	switch (n)
	{
		case 1:
		{
			//console.log("paused="+paused+" running="+running+" testing="+testing );

			if (trainingSetImagesLoaded && !running && !testing && cnn != undefined )
			{
				running = true;
				paused = false;

				btn.innerHTML = "Pause Training";

				if ( epoch == 0 )
				{
					resetGraphForNewNetwork();
				}
				else
				{
					resetGraphForContinuedTraining();
				}
				trainBatchJS();
			}
			else if (running && !paused)
			{
				paused = true;
				btn.innerHTML = "Resume Training";
			}
			else if ( running && paused && !testing )
			{
				paused = false;
				btn.innerHTML = "Pause Training";

				if (timeoutID == undefined)
				{
					trainBatchJS();
				}
			}
			break;
		}

		case 3:
		{

			drawFilters();
			break;
		}

		case 4:
		{
			if ( !running || paused )
			{
				testing = true;
				evaluateTestData();
			}
			break;
		}

		case 7:
		{
			saveWeightsAsJSON( "cnn_"+Date.now()+".json" );
			break;
		}

		case 8: // Load network from JSON
		{
			let fileInput = document.getElementById("networkFile");
			console.log(fileInput);
			fileInput.click();
			break;
		}

		case 0:
		{

			if ( testing )
			{
				break;
			}

			btn.innerHTML = "Start Training";
			running = false;
			paused = false;

			resetGraphForNewNetwork();
			createDefaultNetwork();

			break;
		}
	}
}

function createDefaultNetwork()
{
	cnn = new WebCNN( miniBatchSize );
	cnn.newLayer( { name:"image", type:LAYER_TYPE_INPUT_IMAGE, width:24, height:24, depth:1 } );
	cnn.newLayer( { name:"conv1", type:LAYER_TYPE_CONV, units:10, kernelWidth:5, kernelHeight:5, strideX:1, strideY:1, padding:false} );
	cnn.newLayer( { name:"pool1", type:LAYER_TYPE_MAX_POOL, poolWidth:2, poolHeight:2, strideX:2, strideY:2 } );
	cnn.newLayer( { name:"conv2", type:LAYER_TYPE_CONV, units:20, kernelWidth:5, kernelHeight:5, strideX:1, strideY:1, padding:false} );
	cnn.newLayer( { name:"pool2", type:LAYER_TYPE_MAX_POOL, poolWidth:2, poolHeight:2, strideX:2, strideY:2 } );
	cnn.newLayer( { name:"out",	  type:LAYER_TYPE_FULLY_CONNECTED, units:10, activation:ACTIVATION_SOFTMAX });
	cnn.initialize();

	cnn.setLearningRate( 0.01 );
	cnn.setMomentum( 0.9 );
	cnn.setLambda( 0.0 );
}

function trainBatchJS()
{
	if ( cnn == undefined )
		return;

	if ( !running )
		return;

	timeoutID = undefined;

	if (iterations < totalTrainingImages)
	{
		let imageDataBatch = [];
		let batchLabels = [];

		for ( var example = 0; ( example < miniBatchSize && iterations < totalTrainingImages ); ++example, ++iterations )
		{
			imageDataBatch[ example ] = getTrainingDigitImage( iterations );
			batchLabels[ example ] = digit_labels[ iterations ];
		}

		examplesSeen += miniBatchSize;

		cnn.trainCNNClassifier( imageDataBatch, batchLabels );
		mostRecentTrainingError = cnn.trainingError;

		forwardTime = String( Math.floor( 10 * cnn.forwardTime ) / 10.0 );
		backwardTime = String( Math.floor( 10 * cnn.backwardTime ) / 10.0 );

		if ( forwardTime.indexOf(".") == -1 ) forwardTime += ".0";
		if ( backwardTime.indexOf(".") == -1 ) backwardTime += ".0";

		exampleTimer = Date.now();

		document.getElementById("examplesSeen").innerHTML = String( examplesSeen );


		let prevPlotX = plotX;
		plotX = Math.floor( ( epoch % Math.floor( totalTrainingImages/miniBatchSize ) ) / epochsPerPixel );
		if ( plotX > prevPlotX )
		{
			document.getElementById("forwardTime").innerHTML = forwardTime + " ms";
			document.getElementById("backwardTime").innerHTML = backwardTime + " ms";
			document.getElementById("minibatchLoss").innerHTML = ( Math.floor( 1000.0 * mostRecentTrainingError ) / 1000.0 );
			plotError( mostRecentTrainingError );
			if ( plotX % 10 == 0 )
			{
				const accuracy = getTestDataAccuracy();
				plotAccuracy( accuracy );

				document.getElementById("trainingAccuracy").innerHTML = ( Math.floor( 1000.0 * accuracy ) / 10.0 ) + "%";
			}
		}

		epoch++;

		if (!paused)
		{
			timeoutID = setTimeout( trainBatchJS, 0 );
		}
		else
		{
			console.log( "Pausing after iteration " + iterations );
		}
	}
	else
	{
		running = false;
		paused = false;
		iterations = 0;
		let btn = document.getElementById("startButton");
		btn.innerHTML = "Keep Training";

		const accuracy = getTestDataAccuracy();
		plotAccuracy( accuracy );

		document.getElementById("trainingAccuracy").innerHTML = ( Math.floor( 1000.0 * accuracy ) / 10.0 ) + "%";

		console.log("Done.");
	}
}

function drawFilters()
{
	const newDiv = document.createElement("div");
	document.body.appendChild(newDiv);
	for ( var layerNumber = 1; layerNumber < cnn.layers.length - 1; ++layerNumber )
	{
		let layer = cnn.layers[layerNumber];
		if (layer.type != LAYER_TYPE_CONV) continue;
		console.log("layer "+layer);

		for ( var unit = 0; unit < layer.units; ++unit )
		{
			for ( var kd = 0; kd < layer.kernelDepth; ++kd )
			{
				let canvas = document.createElement( "canvas" );
				canvas.width = layer.kernelWidth;
				canvas.height = layer.kernelHeight;
				let ctx = canvas.getContext( "2d" );
				ctx.mozImageSmoothingEnabled = false;
				ctx.webkitImageSmoothingEnabled = false;
				ctx.msImageSmoothingEnabled = false;
				ctx.imageSmoothingEnabled = false;

				let imageData = ctx.getImageData(0,0,layer.kernelWidth,layer.kernelHeight);
				for ( var ky = 0; ky < layer.kernelHeight; ++ky )
				{
					for (var kx = 0; kx < layer.kernelWidth; ++kx )
					{
						let i = 4 * (layer.kernelWidth * ky + kx);
						let value = 127 + 127 * layer.kernels[ unit ].getValue( kx, ky, kd );
						value =  Math.max( 0, Math.min( 255, value));
						imageData.data[ i ] = value;
						imageData.data[ i + 1 ] = value;
						imageData.data[ i + 2 ] = value;
						imageData.data[ i + 3 ] = 255;
					}
				}

				ctx.putImageData( imageData, 0, 0 );

				let canvas2 = document.createElement( "canvas" );
				canvas2.width = layer.kernelWidth * 5;
				canvas2.height = layer.kernelHeight * 5;
				newDiv.appendChild( canvas2 );
				let ctx2 = canvas2.getContext( "2d" );
				ctx2.mozImageSmoothingEnabled = false;
				ctx2.webkitImageSmoothingEnabled = false;
				ctx2.msImageSmoothingEnabled = false;
				ctx2.imageSmoothingEnabled = false;
				ctx2.drawImage( canvas, 0, 0, canvas2.width, canvas2.height );
			}
		}
	}
}

// Evaluates a random sample of 100 test images, for the
// purpose of plotting approximate test accuracy
function getTestDataAccuracy()
{
	const randomTrialsPerImage = 4;
	const batchSize = 10;
	var correct = 0;
	var imageDataArray = [];
	var labels = [];

	for ( var i = 0; i < testAccuracySampleSize; i += batchSize )
	{
		for ( var example = 0; example < batchSize; ++example )
		{
			var testImageIndex = Math.floor( Math.random() * 10000 );
			var digitLabel = digit_labels[ testImageIndex + 60000 ];
			for ( var randomAugment = 0; randomAugment < randomTrialsPerImage; ++randomAugment )
			{
				imageDataArray[ example * randomTrialsPerImage + randomAugment ] = getTestingDigitImage( testImageIndex );
				labels[ example * randomTrialsPerImage + randomAugment ] = digitLabel;
			}
		}

		const results = cnn.classifyImages( imageDataArray );

		for ( example = 0; example < batchSize; ++example )
		{
			var guess = 0;
			var max = 0;

			for ( var classNum = 0; classNum < results[ example ].dimensions.depth; ++classNum )
			{
				var classSum = 0;
				for ( var randomAugment = 0; randomAugment < randomTrialsPerImage; ++randomAugment )
				{
					classSum += results[ example * randomTrialsPerImage + randomAugment ].getValue( 0, 0, classNum );
				}

				if (  classSum > max )
				{
					max = classSum;
					guess = classNum;
				}
			}

			if ( guess == labels[ example * randomTrialsPerImage ] )
			{
				correct++;
			}
		}
	}

	return ( correct / testAccuracySampleSize );
}

function evaluateTestData()
{
	if ( cnn == undefined )
		return;

	const randomTrialsPerImage = 4;
	const batchSize = 10;
	var correct = 0;
	var imageDataArray = [];
	var labels = [];

	for ( var i = 0; i < 10000; i += batchSize )
	{
		for ( var example = 0; example < batchSize; ++example )
		{
			var digitLabel = digit_labels[ i + 60000 ];
			for ( var randomAugment = 0; randomAugment < randomTrialsPerImage; ++randomAugment )
			{
				imageDataArray[ example * randomTrialsPerImage + randomAugment ] = getTestingDigitImage( i );
				labels[ example * randomTrialsPerImage + randomAugment ] = digitLabel;
			}
		}

		const results = cnn.classifyImages( imageDataArray );

		for ( example = 0; example < batchSize; ++example )
		{
			var guess = 0;
			var max = 0;

			for ( var classNum = 0; classNum < results[ example ].dimensions.depth; ++classNum )
			{
				var classSum = 0;
				for ( var randomAugment = 0; randomAugment < randomTrialsPerImage; ++randomAugment )
				{
					classSum += results[ example * randomTrialsPerImage + randomAugment ].getValue( 0, 0, classNum );
				}

				if (  classSum > max )
				{
					max = classSum;
					guess = classNum;
				}
			}

			if ( guess == labels[ example * randomTrialsPerImage ] )
			{
				correct++;
			}
		}
	}

	console.log( Math.round( correct / 10.0 ) / 10.0 + "%" );
	testing = false;
}

function getTrainingDigitImage( n )
{
	let imageFileIndex = Math.floor( n / 10000 );
	let imageIndex = n % 10000;

	// Add random 0-4 to position for data augmentation
	let y = 28 * Math.floor( imageIndex / 100 ) + Math.floor( Math.random() * 5 );
	let x = 28 * ( imageIndex % 100 )+ Math.floor( Math.random() * 5 );

	return trainingSetCTX[ imageFileIndex ].getImageData( x, y, 24, 24 );
}

function getTestingDigitImage( n )
{
	// Add random 0-4 to position for data augmentation
	let y = 28 * Math.floor( n / 100 ) + Math.floor( Math.random() * 5 );
	let x = 28 * ( n % 100 )+ Math.floor( Math.random() * 5 );

	return testSetCTX.getImageData( x, y, 24, 24 );
}

function saveWeightsAsJSON( filename )
{
	if ( cnn == undefined ) return;

	let networkObj = {};
	networkObj.layers = [];

	//networkObj.miniBatchSize = cnn.getMiniBatchSize();
	networkObj.examplesSeen = examplesSeen;
	networkObj.miniBatchSize = miniBatchSize;
	networkObj.momentum = cnn.getMomentum();
	networkObj.learningRate = cnn.getLearningRate();
	networkObj.lambda = cnn.getLambda();

	for ( var layerIndex = 0; layerIndex < cnn.layers.length; ++layerIndex )
	{
		let layer = cnn.layers[ layerIndex ];

		let layerObj = {};
		layerObj.name = layer.name;
		layerObj.type = layer.type;
		layerObj.index = layerIndex;

		switch ( layer.type )
		{
			case LAYER_TYPE_INPUT_IMAGE:
			{
				layerObj.width = layer.outputDimensions.width;
				layerObj.height = layer.outputDimensions.height;
				layerObj.depth = layer.outputDimensions.depth;
				break;
			}

			case LAYER_TYPE_FULLY_CONNECTED:
			{
				layerObj.units = layer.units;
				layerObj.weights = [];
				layerObj.activation =  layer.activation;
				for ( var unit = 0; unit < layer.units; ++unit )
				{
					layerObj.weights[unit] = layer.weights[ unit ].getValuesAsArray();
					layerObj.biases = Array.from( layer.biases );
				}
				break;
			}

			case LAYER_TYPE_CONV:
			{
				layerObj.units = layer.units;
				layerObj.weights = [];
				layerObj.kernelWidth = layer.kernelWidth;
				layerObj.kernelHeight = layer.kernelHeight;
				layerObj.strideX = layer.strideX;
				layerObj.strideY = layer.strideY;
				layerObj.padX = layer.padX;
				layerObj.padY = layer.padY;

				for ( var unit = 0; unit < layer.units; ++unit )
				{
					layerObj.weights[unit] = layer.kernels[ unit ].getValuesAsArray();
					layerObj.biases = Array.from( layer.biases );
				}

				break;
			}

			case LAYER_TYPE_MAX_POOL:
			{
				layerObj.poolWidth = layer.poolWidth;
				layerObj.poolHeight = layer.poolHeight;
				layerObj.strideX = layer.strideX;
				layerObj.strideY = layer.strideY;
				break;
			}
		}

		networkObj.layers[ layerIndex ] = layerObj;
	}

	let json = JSON.stringify(networkObj, null, "\t");
	let a = document.createElement('a');
	a.setAttribute('href', 'data:text/plain;charset=utf-u,'+encodeURIComponent(json));
	a.setAttribute('download', filename);
	a.click();
}

function OnNetworkJSONLoaded(evt)
{
	console.log("parsing");
	let networkFileText = evt.target.result;
	let networkJSON = JSON.parse( networkFileText );
	loadNetworkFromJSON( networkJSON );
}

function loadNetworkFromJSON( networkJSON )
{
	cnn = new WebCNN();

	if (networkJSON.momentum != undefined) cnn.setMomentum( networkJSON.momentum );
	if (networkJSON.lambda != undefined) cnn.setLambda( networkJSON.lambda );
	if (networkJSON.learningRate != undefined) cnn.setLearningRate( networkJSON.learningRate );

	// Note that this parameter is not a property of the network, but of the training function in this script
	if (networkJSON.miniBatchSize != undefined) miniBatchSize = networkJSON.miniBatchSize;

	for ( var layerIndex = 0; layerIndex < networkJSON.layers.length; ++layerIndex )
	{
		let layerDesc = networkJSON.layers[ layerIndex ];
		console.log( layerDesc );
		cnn.newLayer( layerDesc );
	}

	for ( var layerIndex = 0; layerIndex < networkJSON.layers.length; ++layerIndex )
	{
		let layerDesc = networkJSON.layers[ layerIndex ];

		switch ( networkJSON.layers[ layerIndex ].type )
		{
			case LAYER_TYPE_CONV:
			case LAYER_TYPE_FULLY_CONNECTED:
			{
				if (layerDesc.weights != undefined && layerDesc.biases != undefined )
				{
					cnn.layers[ layerIndex ].setWeightsAndBiases( layerDesc.weights, layerDesc.biases );
				}
				break;
			}
		}
	}

	cnn.initialize();
}