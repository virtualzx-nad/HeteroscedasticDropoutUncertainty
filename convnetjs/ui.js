
var funcForm="Math.pow(1-Math.exp(2-x), 2)";
var scalingForm="Math.exp(1-x)";
var layerTypes = ["elu"];
var layerUnits = ["10"];
var allowedTypes=["elu", "relu", "tanh","sigmoid","linear"]

function setTrainer(method, learning_rate, momentum){
  trainer = new convnetjs.Trainer(net, {
                            method:method, learning_rate:learning_rate, momentum:momentum,
                            batch_size:10, l2_decay:l2_decay});
}


function setNetworkOnClick(e){
  // Set Network Structure button is clicked.
  dropRateInput = document.getElementById('dropOutRate').value;
  l2_decay = l2 * (1 - dropRateInput) / (2 * N);
  //hetero = document.getElementById('hetero').checked;
  createLayers(dropRateInput);
  reload_reg();
}



function setOptimizerOnClick(e){
  optMethod = document.getElementById('methodRadio').value;
  mom = document.getElementById('sgdMomentum').value;
  lnRate = document.getElementById('learningRate').value;
  //window.alert("method:"+optMethod+", mom:"+mom+", lnRate:"+lnRate);
  setTrainer(optMethod, lnRate, mom);
}


function createLayers(dropoutRate){
  // create neural net
  layer_defs = [];
  layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:1});
  for(var i=0;i<layerTypes.length;i++){
      if(i!=0) layer_defs.push({type:'dropout', drop_prob:dropoutRate});
      layer_defs.push({type:'fc', num_neurons:layerUnits[i], activation:layerTypes[i]});    
  }
  layer_defs.push({type:'heteroscedastic_regression', num_neurons:2});
  console.log('layers:'+layerTypes+', units:'+layerUnits);
}


function setFuncForm(e){
	funcForm = document.getElementById('funcForm').value;
  scalingForm = document.getElementById('scalingForm').value;
  regen_data();
  reload_reg();
}

function updateLayerType(me){
    var row=me.parentNode.parentNode;
    var i=row.rowIndex;
    var newLayerType = me.value.toLowerCase();
    if (!allowedTypes.includes(newLayerType)){
        window.alert("Invalid layer type:"+newLayerType);
        me.value = me.oldvalue;
        return;
    }
    layerTypes[i-1] = newLayerType;
}

function updateLayerUnits(me){
    var row=me.parentNode.parentNode;
    var i=row.rowIndex;
    var newLayerUnits = Math.round(me.value);
    if (isNaN(newLayerUnits)||newLayerUnits<1||newLayerUnits>1000){
        window.alert("Invalid layer units:"+newLayerUnits);
        me.value = me.oldvalue;
        return;
    }
    layerUnits[i-1] = newLayerUnits;
}

function addDeleteRow(me)
{
    var row=me.parentNode.parentNode;
    var i=row.rowIndex;
    mytable=document.getElementById('layerTable');
    var nrows = mytable.rows.length;
    if(i==nrows-1){
        var inputs = row.getElementsByTagName('input');
        var newLayerType;
        var newLayerUnits;
        for (var i=0, iLen=inputs.length; i<iLen; i++) {
            if(inputs[i].id=="myacttype"){
                newLayerType = inputs[i].value.toLowerCase();
                if (!allowedTypes.includes(newLayerType)){
                    window.alert("Invalid layer type:"+newLayerType);
                    return;
                }
            } 
            if(inputs[i].id=="myunits"){
                newLayerUnits = Math.round(inputs[i].value);
                if (isNaN(newLayerUnits)||newLayerUnits<1||newLayerUnits>1000){
                    window.alert("Invalid unit number:"+newLayerUnits);
                    return;
                }
            } 
        }
        layerTypes.push(newLayerType);
        layerUnits.push(newLayerUnits);
        var new_row = row.cloneNode(true);
        me.value = "Delete";
        mytable.appendChild( new_row );
    }
    else{
        mytable.deleteRow(i);
        layerTypes.splice(i-1, 1);
        layerUnits.splice(i-1, 1);
    }

}
