// Tunable parameters:
var N = 15;
var dropoutRate = 0.05;
var l2 = 0.01; // p_l(w) = N(w; 0, l^{−2}I); l^{−2} = 10

var l2_decay = l2 * (1 - dropoutRate) / (2 * N);
console.log('l2_decay = ' + l2_decay);

var data, labels;
var ntest = 50;
var test_x;
var true_values;
var density = 5.0;
var x_start=-1; 
var x_end=4;
var ss = 30.0; // scale for drawing
var acc = 0;

var layer_defs, net, trainer, sum_y, sum_y_sq, sum_sigma2;

// create neural net
layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:1});
// layer_defs.push({type:'dropout', drop_prob:p}); // this is not a good idea when we have a one dimensional input!
layer_defs.push({type:'fc', num_neurons:10, activation:'elu'}); // num_neurons = num of outputs
layer_defs.push({type:'dropout', drop_prob:dropoutRate});
layer_defs.push({type:'heteroscedastic_regression', num_neurons:2}); // this layer always adds one more fc layer

var lix=2; // layer id of layer we'd like to draw outputs of
function reload_reg() {
  net = new convnetjs.Net();
  net.makeLayers(layer_defs);

  trainer = new convnetjs.Trainer(net, {learning_rate:0.01, momentum:0.3, batch_size:12, l2_decay:l2_decay});

  sum_y = Array();
  for(var x=0.0; x<=WIDTH; x+= density)
    sum_y.push(new cnnutil.Window(100, 0));
  sum_y_sq = Array();
  for(var x=0.0; x<=WIDTH; x+= density)
    sum_y_sq.push(new cnnutil.Window(100, 0));
  sum_sigma2 = Array();
  for(var x=0.0; x<=WIDTH; x+= density)
    sum_sigma2.push(new cnnutil.Window(100, 0));  
  test_x = Array();
  true_values = Array();
  var x1, x2;
  x1 = x_start - (x_end-x_start)/10;
  x2 = x_end + (x_end-x_start)/10;
  var dx = (x2-x1)/(ntest-1);
  for(var i=0; i<ntest; i++){
    var x = x1 + i * dx;
    var y = eval(funcForm);
    test_x.push(x);
    true_values.push(y);
  }

  acc = 0;
}
 
function regen_data() {
  sum_y = Array();
  for(var x=0.0; x<=WIDTH; x+= density)
    sum_y.push(new cnnutil.Window(100, 0));
  sum_y_sq = Array();
  for(var x=0.0; x<=WIDTH; x+= density)
    sum_y_sq.push(new cnnutil.Window(100, 0));
  sum_sigma2 = Array();
  for(var x=0.0; x<=WIDTH; x+= density)
    sum_sigma2.push(new cnnutil.Window(100, 0));
  test_x = Array();
  true_values = Array();
  var x1, x2;
  x1 = x_start - (x_end-x_start)/5;
  x2 = x_end + (x_end-x_start)/5;
  var dx = (x2-x1)/(ntest-1);
  for(var i=0; i<ntest; i++){
    var x = x1 + i * dx;
    var y = eval(funcForm);
    test_x.push(x);
    true_values.push(y);
  }
  acc = 0;
  data = [];
  labels = [];
  for(var i=0;i<N;i++) {
    var x = Math.random()*(x_end-x_start)+x_start;
    var y = eval(funcForm); 
    data.push([x]);
    labels.push([y]);
  }
}

function myinit(){
  regen_data();
  reload_reg();
}
 
function update_reg(){
  // forward prop the data
  
  var netx = new convnetjs.Vol(1,1,1);
  avloss = 0.0;

  for(var iters=0;iters<50;iters++) {
    for(var ix=0;ix<N;ix++) {
      netx.w = data[ix];
      var stats = trainer.train(netx, labels[ix]);
      avloss += stats.loss;
    }
  }
  avloss /= N*iters;

}

function draw_reg(){    
    ctx_reg.clearRect(0,0,WIDTH,HEIGHT);
    ctx_reg.fillStyle = "black";

    var netx = new convnetjs.Vol(1,1,1);

    // draw decisions in the grid
    var draw_neuron_outputs = $("#layer_outs").is(':checked');
    
    // draw final decision
    var neurons = [];
    ctx_reg.globalAlpha = 0.5;
    ctx_reg.beginPath();
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {

      netx.w[0] = (x-WIDTH/2)/ss;
      var a = net.forward(netx);
      var y = a.w[0];
      sum_y[c].add(y);
      sum_y_sq[c].add(y*y);
      var ls2 = a.w[1];
      var sigma2 = Math.exp(ls2)
      // we need to average the sigma2 samples following the same derivations for sum_y_sq.
      sum_sigma2[c].add(sigma2);

      if(draw_neuron_outputs) {
        neurons.push(net.layers[lix].out_act.w); // back these up
      }

      if(x===0) ctx_reg.moveTo(x, -y*ss+HEIGHT/2);
      else ctx_reg.lineTo(x, -y*ss+HEIGHT/2);
      c += 1;
    }
    // console.log('last ls2 = ' + ls2);
    // console.log('bias[1] = ' + net.layers[net.layers.length - 1].biases.w[1]);
    // console.log('Ws[0] = ' + net.layers[net.layers.length - 1].filters[1].w[0]);
    // console.log('sum_sigma2[0].get_average() = ' + sum_sigma2[0].get_average()); // far to the left
    // console.log('sum_sigma2[140].get_average() = ' + sum_sigma2[140].get_average()); // far to the right

    acc += 1;
    ctx_reg.stroke();
    ctx_reg.globalAlpha = 1.;

    // draw individual neurons on first layer
    if(draw_neuron_outputs) {
      var NL = neurons.length;
      ctx_reg.strokeStyle = 'rgb(250,50,50)';
      for(var k=0;k<NL;k++) {
        ctx_reg.beginPath();
        var n = 0;
        for(var x=0.0; x<=WIDTH; x+= density) {
          if(x===0) ctx_reg.moveTo(x, -neurons[n][k]*ss+HEIGHT/2);
          else ctx_reg.lineTo(x, -neurons[n][k]*ss+HEIGHT/2);
          n++;
        }
        ctx_reg.stroke();
      }
    }
  
    // draw axes
    ctx_reg.beginPath();
    ctx_reg.strokeStyle = 'rgb(50,50,50)';
    ctx_reg.lineWidth = 1;
    ctx_reg.moveTo(0, HEIGHT/2);
    ctx_reg.lineTo(WIDTH, HEIGHT/2);
    ctx_reg.moveTo(WIDTH/2, 0);
    ctx_reg.lineTo(WIDTH/2, HEIGHT);
    ctx_reg.stroke();

    // draw datapoints. Draw support vectors larger
    ctx_reg.strokeStyle = 'rgb(0,0,0)';
    ctx_reg.lineWidth = 1;
    for(var i=0;i<N;i++) {
      drawCircle(data[i]*ss+WIDTH/2, -labels[i]*ss+HEIGHT/2, 5.0);
    }    

    // Draw true function value
    ctx_reg.beginPath();
    ctx_reg.lineWidth = 2;
    ctx_reg.strokeStyle = 'rgb(0,0,0)';
    var error = 0.0;
    for(var i=0; i<ntest; i++) {
      x = test_x[i]*ss+WIDTH/2;
      y = -true_values[i]*ss+HEIGHT/2;
      netx.w[0] = x;
      var a = net.forward(netx);
      prediction = a.w[0];
      error += Math.pow(prediction - true_values[i], 2)
      if(i===0) ctx_reg.moveTo(x, y);
      else ctx_reg.lineTo(x, y);
    }
    error = Math.sqrt(error / ntest);
    ctx_reg.stroke();

    // Draw the mean plus minus 2 standard deviations
    ctx_reg.beginPath();
    ctx_reg.lineWidth = 1;
    ctx_reg.strokeStyle = 'rgb(0,0,250)';
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y[c].get_average();
      if(x===0) ctx_reg.moveTo(x, -mean*ss+HEIGHT/2);
      else ctx_reg.lineTo(x, -mean*ss+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();
    // Draw the uncertainty
    ctx_reg.fillStyle = 'rgb(0,0,250)';
    ctx_reg.globalAlpha = 0.2;
    ctx_reg.beginPath();
    var c = 0;
    var start = 0
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y[c].get_average();
      var std = Math.sqrt(sum_y_sq[c].get_average() - mean * mean + sum_sigma2[c].get_average());
      mean += std * 2;
      pos = Math.min(Math.max(-mean*ss+HEIGHT/2, 0), HEIGHT);
      if(x===0) {start = pos; ctx_reg.moveTo(x, start); }
      else ctx_reg.lineTo(x, pos);
      c += 1;
    }
    var c = sum_y.length - 1;
    for(var x=WIDTH; x>=0.0; x-= density) {
      var mean = sum_y[c].get_average();
      var std = Math.sqrt(sum_y_sq[c].get_average() - mean * mean + sum_sigma2[c].get_average());
      mean -= std * 2;
      pos = Math.min(Math.max(-mean*ss+HEIGHT/2, 0), HEIGHT);
      ctx_reg.lineTo(x, pos);
      c -= 1;
    }
    ctx_reg.lineTo(0, start);
    ctx_reg.fill();

    // Draw the aleatoric uncertainty  
    ctx_reg.fillStyle = 'rgb(250,0,0)';
    ctx_reg.globalAlpha = 1.;
    ctx_reg.globalAlpha = 0.2;
    ctx_reg.beginPath();
    var c = 0;
    var start = 0
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y[c].get_average();
      var std = Math.sqrt(sum_sigma2[c].get_average());
      mean += std * 2;
      pos = Math.min(Math.max(-mean*ss+HEIGHT/2, 0), HEIGHT);
      if(x===0) {start = pos; ctx_reg.moveTo(x, start); }
      else ctx_reg.lineTo(x, pos);
      c += 1;
    }
    var c = sum_y.length - 1;
    for(var x=WIDTH; x>=0.0; x-= density) {
      var mean = sum_y[c].get_average();
      var std = Math.sqrt(sum_sigma2[c].get_average());
      mean -= std * 2;
      pos = Math.min(Math.max(-mean*ss+HEIGHT/2, 0), HEIGHT);
      ctx_reg.lineTo(x, pos);
      c -= 1;
    }
    ctx_reg.lineTo(0, start);
    ctx_reg.fill();


    ctx_reg.strokeStyle = 'rgb(0,0,0)';
    ctx_reg.globalAlpha = 1.;

    ctx_reg.fillStyle = "blue";
    ctx_reg.font = "bold 16px Arial";
    ctx_reg.fillText("average loss: " + avloss, 20, 15);
    ctx_reg.fillStyle = "red";
    ctx_reg.font = "bold 16px Arial";
    ctx_reg.fillText("RMSE: " + error, 20, 35);
}

// function addPoint(x, y){
//   // add datapoint at location of click
//   alert($(NPGcanvas).width())
//   data.push([(x-$(NPGcanvas).width()/2)/ss]);
//   labels.push([-(y-$(NPGcanvas).height()/2)/ss]);
//   N += 1;
// }

function mouseClick(x, y, shiftPressed){  
  // add datapoint at location of click
  // alert(WIDTH);
  // alert($(NPGcanvas).width());
  // alert(ss);
  //alert(x);
  x = x / $(NPGcanvas).width() * WIDTH;
  y = y / $(NPGcanvas).height() * HEIGHT;
  data.push([(x-WIDTH/2)/ss]);
  labels.push([-(y-HEIGHT/2)/ss]);
  N += 1;
}