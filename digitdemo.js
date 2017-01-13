let cnn;
window.onload = Init;

let drawCanvas;
let ctx_draw;
let drawing = false;
let lastPos;
let lineWidth = 28;
let drawingPathIndex = -1;
let drawingPaths = [];
const TAU = Math.PI * 2;
let freshCanvas = true;

function Init()
{
	drawCanvas = document.getElementById("drawCanvas");
	ctx_draw = drawCanvas.getContext("2d");
	ctx_draw.mozImageSmoothingEnabled = false;
	ctx_draw.webkitImageSmoothingEnabled = false;
	ctx_draw.msImageSmoothingEnabled = false;
	ctx_draw.imageSmoothingEnabled = false;

	resetDrawingCanvas();

	drawCanvas.addEventListener("mousedown",onMouseDown,false);
	window.addEventListener("mouseup",onMouseUp,false);
	drawCanvas.addEventListener("mousemove",onMouseMove,false);
	drawCanvas.addEventListener("contextmenu",onContextMenu,false);
	drawCanvas.addEventListener("mouseout",onMouseOut,false);
	drawCanvas.addEventListener("mouseover",onMouseOver,false);

	// Load the network
	$.ajax({
		url: "cnn_mnist_10_20_98accuracy.json",
		dataType: "json",
		success: onJSONLoaded
	});
}

function resetDrawingCanvas()
{
	if ( ctx_draw == undefined )
		return;

	freshCanvas = true;
	ctx_draw.fillStyle = "white";
	ctx_draw.fillRect( 0, 0, drawCanvas.width, drawCanvas.height );
	ctx_draw.fillStyle = "rgb(200,200,200)";
	ctx_draw.font = "22px Verdana";
	ctx_draw.fillText("Draw a digit (0-9) here", 24, 150);
}

function onJSONLoaded( response )
{
	loadNetworkFromJSON( response );
	console.log("JSON Loaded!");
}

function onContextMenu(e)
{
	e.preventDefault();
}

function onMouseDown(e)
{
	if ( freshCanvas )
	{
		freshCanvas = false;
		ctx_draw.fillStyle = "white";
		ctx_draw.fillRect( 0, 0, drawCanvas.width, drawCanvas.height );
	}
	drawing = true;
	drawingPathIndex++;
	drawingPaths[ drawingPathIndex ] = [];
	lastPos = [ e.offsetX, e.offsetY ];
	ctx_draw.strokeStyle = "black";
	ctx_draw.fillStyle = "black";
	ctx_draw.lineCap = "round";
	ctx_draw.lineJoin = "round";
	ctx_draw.beginPath();
	ctx_draw.arc( e.offsetX, e.offsetY, lineWidth / 2, 0, TAU );
	ctx_draw.fill();

}

function onMouseUp(e)
{
	if (drawing)
	{
		guessNumber();
		drawing = false;
		lastPos = undefined;
	}
}

function onMouseOut(e)
{
	drawing = false;
	lastPos = undefined;
}

function onMouseOver(e)
{

}

function onMouseMove(e)
{

	if ( !drawing ) return;
	if ( e.target != drawCanvas )
	{
		drawing = false;
		lastPos = undefined;
		return;
	}

	var x = Math.max( 0, Math.min( e.target.width, e.offsetX ) );
	var y = Math.max( 0, Math.min( e.target.height, e.offsetY ) );

	if ( e.offsetX > 0 && e.offsetX < e.target.width && e.offsetY > 0 && e.offsetY < e.target.height )
	{
		ctx_draw.lineWidth = lineWidth;

		if ( lastPos != undefined )
		{
			//ctx_draw.beginPath();
			//ctx_draw.arc( x, y, 14, 0, TAU);
			//ctx_draw.fill();
			ctx_draw.beginPath();
			ctx_draw.moveTo( lastPos[ 0 ], lastPos[ 1 ] );
			ctx_draw.lineTo( x, y );
			ctx_draw.stroke();
		}
		else
		{
			drawingPathIndex++;
			ctx_draw.beginPath();
			ctx_draw.arc( x, y, lineWidth / 2, 0, TAU );
			ctx_draw.fill();
		}

		if ( drawingPaths[ drawingPathIndex ] == undefined )
		{
			drawingPaths[ drawingPathIndex ] = [];
		}

		drawingPaths[ drawingPathIndex ].push( [ x, y ] );
		lastPos = [ x, y ];
	}
	else
	{
		lastPos = undefined;
	}
}

function buttonClick( n )
{
	switch (n)
	{
		case 1:
		{
			guessNumber();
			break;
		}

		case 2:
		{
			drawingPaths = [];
			drawingPathIndex = -1;
			lastPos = undefined;

			document.getElementById("guessNumberDiv").innerHTML = "";
			document.getElementById("confidence").innerHTML = "";

			resetDrawingCanvas();
			break;
		}
	}
}

function preProcessDrawing()
{
	let drawnImageData = ctx_draw.getImageData( 0, 0, ctx_draw.canvas.width, ctx_draw.canvas.height );

	var xmin = ctx_draw.canvas.width - 1;
	var xmax = 0;
	var ymin = ctx_draw.canvas.height - 1;
	var ymax = 0;
	var w = ctx_draw.canvas.width;
	var h = ctx_draw.canvas.height;

	// Find bounding rect of drawing
	for ( let i = 0; i < drawnImageData.data.length; i+=4 )
	{
		var x = Math.floor( i / 4 ) % w;
		var y = Math.floor( i / ( 4 * w ) );

		if ( drawnImageData.data[ i ] < 255 || drawnImageData.data[ i + 1 ] < 255 || drawnImageData.data[ i + 2 ] < 255 )
		{
			xmin = Math.min( xmin, x );
			xmax = Math.max( xmax, x );
			ymin = Math.min( ymin, y );
			ymax = Math.max( ymax, y );
		}
	}

	const cropWidth = xmax - xmin;
	const cropHeight = ymax - ymin;

	if ( cropWidth > 0 && cropHeight > 0 && ( cropWidth < w || cropHeight < h ) )
	{
		// Crop and scale drawing
		const scaleX = cropWidth  / w;
		const scaleY = cropHeight / h;
		const scale = Math.max( scaleX, scaleY );
		const scaledLineWidth = Math.max( 1, Math.floor( lineWidth * scale ) );
		//console.log(scale);

		// Scaling down, redraw image with scale lineWidth
		const tempCanvas = document.createElement("canvas");

		//document.body.appendChild(tempCanvas);
		tempCanvas.width = w;
		tempCanvas.height = h;
		const ctx_temp = tempCanvas.getContext("2d");

		ctx_temp.strokeStyle = "black";
		ctx_temp.fillStyle = "black";
		ctx_temp.lineCap = "round";
		ctx_temp.lineJoin = "round";
		ctx_temp.lineWidth = scaledLineWidth;

		//console.log(drawingPaths);

		//console.log(drawingPaths);
		for ( var pathIndex = 0; pathIndex < drawingPaths.length; ++pathIndex )
		{
			var path = drawingPaths[ pathIndex ];
			if ( path == undefined || path.length == 0)
			{
				continue;
			}
			var p = path[0];
			ctx_temp.beginPath();
			ctx_temp.moveTo( p[0], p[1] );

			for ( var i = 1; i < path.length; ++i )
			{
				p = path[ i ];
				ctx_temp.lineTo( p[0], p[1] );
			}
			ctx_temp.stroke();
		}

		drawnImageData = ctx_temp.getImageData( xmin, ymin, cropWidth, cropHeight );
	}

	// Invert black and white to match training data
	for ( var i = 0; i < drawnImageData.data.length; i+=4 )
	{
		drawnImageData.data[i] = 255 - drawnImageData.data[i];
		drawnImageData.data[i+1] = 255 - drawnImageData.data[i+1];
		drawnImageData.data[i+2] = 255 - drawnImageData.data[i+2];
	}

	let canvas2 = document.createElement( "canvas" );
	canvas2.width = drawnImageData.width;
	canvas2.height = drawnImageData.height;
	//document.body.appendChild(canvas2);
	let ctx2 = canvas2.getContext( "2d" );
	ctx2.mozImageSmoothingEnabled = false;
	ctx2.webkitImageSmoothingEnabled = false;
	ctx2.msImageSmoothingEnabled = false;
	ctx2.imageSmoothingEnabled = false;
	ctx2.putImageData( drawnImageData, 0, 0 );

	let canvas = document.createElement( "canvas" );
	canvas.width = 24;
	canvas.height = 24;
	//document.body.appendChild(canvas);
	let ctx = canvas.getContext( "2d" );
	ctx.mozImageSmoothingEnabled = false;
	ctx.webkitImageSmoothingEnabled = false;
	ctx.msImageSmoothingEnabled = false;
	ctx.imageSmoothingEnabled = false;

	// Preserve aspect ratio of cropped section, center it

	var xOffset = 0;
	var yOffset = 0;
	var xScale = 1;
	var yScale = 1;
	const padding = 1;

	if ( canvas2.width > canvas2.height )
	{
		yOffset = ( canvas.width / ( canvas2.width + 2 * padding) ) * ( canvas2.width - canvas2.height ) / 2 + padding;
		yScale = canvas2.height / canvas2.width;

		xOffset = padding;

	}
	else if ( canvas2.height > canvas2.width )
	{
		xOffset = ( canvas.height / canvas2.height ) * ( canvas2.height - canvas2.width ) / 2 + padding;
		xScale = canvas2.width / canvas2.height;

		yOffset = padding;
	}

	ctx.fillStyle = "black";
	ctx.fillRect( 0, 0, canvas.width, canvas.height );
	ctx.drawImage( canvas2, xOffset, yOffset, canvas.width * xScale - 2 * padding, canvas.height * yScale - 2 * padding );

	return ctx.getImageData( 0, 0, 24, 24 );
}

function guessNumber()
{
	const inputImageData = preProcessDrawing();

	if ( cnn == undefined ) return;

	const result = cnn.classifyImages( [ inputImageData ] );

	let guess = 0;
	let max = 0;
	for ( var i = 0; i < 10; ++i )
	{
		if ( result[ 0 ].getValue( 0, 0, i ) > max )
		{
			max = result[ 0 ].getValue( 0, 0, i );
			guess = i;
		}
	}
	//console.log("Is it a "+guess+" ? ("+Math.floor( 1000 * max ) / 10.0 + "%");

	document.getElementById("guessNumberDiv").innerHTML = ( max > 0.666667 ) ? String( guess ) : "?";
	document.getElementById("confidence").innerHTML = String( Math.min( 100, Math.floor( 1000 * ( max + 0.1 ) ) / 10.0 ) ) + "% it's a "+String( guess );
}

function loadNetworkFromJSON( networkJSON )
{
	cnn = new WebCNN();

	if (networkJSON.momentum != undefined) cnn.setMomentum( networkJSON.momentum );
	if (networkJSON.lambda != undefined) cnn.setLambda( networkJSON.lambda );
	if (networkJSON.learningRate != undefined) cnn.setLearningRate( networkJSON.learningRate );

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

function OnNetworkJSONLoaded(evt)
{
	console.log("parsing");
	let networkFileText = evt.target.result;

	let networkJSON = JSON.parse( networkFileText );

	loadNetworkFromJSON( networkJSON );


}