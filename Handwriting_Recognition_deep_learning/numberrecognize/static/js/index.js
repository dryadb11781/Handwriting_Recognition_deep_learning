var counter=0;
var mark_label=0;//mark值
var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');
var canvas2 = document.getElementById('myCanvas2');
var ctx2 = canvas2.getContext('2d');
var resultcsv = [];
var user_name;
var zero=0,one=0,two=0,three=0,four=0,five=0,six=0,seven=0,eight=0,nine=0;
function mouseDown(e){
    this.draw=true;
    this.ctx = this.getContext("2d");
    this.ctx.strokeStyle='#010101';
    this.ctx.lineWidth=5;

    var o=this;
    this.offsetX=this.offsetLeft;
    this.offsetY=this.offsetTop;

    while(o.offsetParent){
    	o=o.offsetParent;
    	this.offsetX+=o.offsetLeft;
    	this.offsetY+=o.offsetTop;
    }

    this.ctx.beginPath();
    this.ctx.moveTo(e.pageX-this.offsetX,e.pageY-this.offsetY);

}

function mouseMove(e){
    if (this.draw){
        this.ctx.lineTo(e.pageX-this.offsetX,e.pageY-this.offsetY);
        this.ctx.stroke();
    }
}

function mouseUp(e){
     ctx.save();
    this.draw=false;


  }



function clearPad(){
    var canvas=document.querySelector('#myCanvas');
    var ctx=canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx2.clearRect(0,0,canvas.width,canvas.height);


}

function gray_level_piex_acces(){

  //rescale and redraw to save cavas state
  ctx2.drawImage(canvas, 0, 0, canvas.width,canvas.height,0,0,canvas2.height,canvas2.height);
  var imageData2 = ctx2.getImageData(0, 0, canvas2.width, canvas2.height);
  var data2 = imageData2.data;
  var image_array= new Array();

  //image_array.push(mark_label);
  for (var i = 0; i < data2.length; i += 4) {
    if(data2[i]!=0){
    data2[i] =255;
    }
    image_array.push(data2[i]);//save this number
  }
//console.log( JSON.stringify(image_array) );
predict_result_label = document.getElementById('number_result');
$.get('/predict/?test='+image_array.toString(), function(data, status){predict_result_label.textContent=data});




image_array=[];
clearPad();
//label_hint=document.getElementById('label_hint');


}


window.addEventListener('load',function(){
    var canvas=document.querySelector('#myCanvas');

    canvas.addEventListener('mousedown',mouseDown);
    canvas.addEventListener('mousemove',mouseMove);
    canvas.addEventListener('mouseup',mouseUp);


});
//
