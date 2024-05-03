//extentions : allow user to upload images to use
import {vec4,mat4,mat3} from 'https://webgpufundamentals.org/3rdparty/wgpu-matrix.module.js';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import { mergeSort } from './merge.js';
import {test} from './radix_sort_m.js';
import { EngineContext } from "./EngineContext.js"
  
function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}

function splitList(inputList,chunks) {
  const totalLength = Math.ceil(inputList.length/chunks);
  const sublistLength = Math.ceil(totalLength / 4);
  const result = [];

  for (let i = 0; i < 4; i++) {
    const startIdx = i * sublistLength;
    const endIdx = startIdx + sublistLength;
    result.push(inputList.slice(startIdx*chunks, endIdx*chunks));
  }

  return result;
}

function chunkArray(inputArray, chunkSize) {
  let res = [];
  for (let i = 0; i < inputArray.length; i += chunkSize) {
    res.push(inputArray.slice(i, i + chunkSize));
  }
  return res;
}

async function main() {

  //initialize devices and check that the browser supports webgpu
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!device) {
    fail('need a browser that supports WebGPU');
    return;
  }


  // Get a WebGPU context from the canvas and configure it
  const canvas = document.querySelector('canvas');
  canvas.width = 1280;
  canvas.height = 720;
  const context = canvas.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
  });
  
  //Load shader from shaders/shader.txt
  var module = device.createShaderModule({
    label: 'our hardcoded textured quad shaders',
    code: await (await fetch("src/shaders/shader.txt")).text(),
  });

  //initialize the rendering pipeline, with alpha blending.
  const RenderPipeline = device.createRenderPipeline({
    label: 'hardcoded textured quad pipeline',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs',
      buffers: [
        // keys
        {
          arrayStride: 4,
          stepMode: 'instance',
          attributes: [
            {shaderLocation: 0, offset: 0, format: 'sint32' },
          ],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [{ format: presentationFormat, blend: {color: {srcFactor : "one-minus-src-alpha", dstFactor: "dst-alpha", operation:"add"}, alpha: {srcFactor : "src-alpha", dstFactor: "dst-alpha", operation : "add"}} }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });
  

  //initialize a renderPassDescriptor for the pipeline to use
  const renderPassDescriptor = {
    label: 'our basic canvas renderPass',
    colorAttachments: [
      {
        // view: <- to be filled out when we render
        loadOp: 'load',
        storeOp: 'store',
      },
    ],
  };
  

  const material = new THREE.PointsMaterial( { size: 0.01, vertexColors: true } );
  var cloud;
  const loader = new PLYLoader();
  loader.setCustomPropertyNameMapping({
    pointsColour: ['x','y','z','nx','ny','nz','f_rest_0','f_rest_1','f_rest_2','opacity'],
    scaleRotation: ['scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'],
    //Color: ['f_dc_0','f_dc_1','f_dc_2','f_rest_0','f_rest_1','f_rest_2','f_rest_3','f_rest_4','f_rest_5','f_rest_6','f_rest_7','f_rest_8'],
    Color: ['f_dc_0','f_dc_1','f_dc_2'],
    Color2: ['f_rest_9','f_rest_10','f_rest_11','f_rest_12','f_rest_13','f_rest_14','f_rest_15','f_rest_16','f_rest_17','f_rest_18','f_rest_19','f_rest_20','f_rest_21','f_rest_22','f_rest_23'],
    //Color3: ['f_rest_24','f_rest_25','f_rest_26','f_rest_27','f_rest_28','f_rest_29','f_rest_30','f_rest_31','f_rest_32','f_rest_33','f_rest_34','f_rest_35','f_rest_36','f_rest_37','f_rest_38','f_rest_39','f_rest_40','f_rest_41','f_rest_42','f_rest_43','f_rest_44'],
    //17 elements
    //f_dc TAKES VALUES > 1? IS THIS THE COLOUR CHANNEL?? OPACITY ALSO TAKES NEGATIVE VALUES AND VALUES > 1, MAYBE FINE DUE TO THE VALUES RETURNED BY GAUSSIAN FUNCTION.
  })

  var data;
  var pointsColourData;
  var covarianceMatrices;
  var ColorData;
  var ColorData2;
  var scaleRotationData;

  function redraw(){
    initBuffers(pointsColourData,covarianceMatrices,ColorData,ColorData2);
    compute(covarianceMatrices,pointsColourData,camera,ColorData,ColorData2,scaleRotationData);
  }

  document.addEventListener('keydown', function(event) {
    if(event.keyCode == 37) {
      camera.x += 1;
      redraw();

    }
    else if(event.keyCode == 38) {
        camera.y -= 1;
        redraw();
    }
    else if(event.keyCode == 37) {
      camera.x += 1;
      redraw();

    }
    else if(event.keyCode == 40) {
      camera.y += 1;
      redraw();
    }
    else if(event.keyCode == 78) {
      camera.z += 1;
      redraw();

    }
    else if(event.keyCode == 77) {
      camera.z -= 1;
      redraw();
    }
    else if(event.keyCode == 68) {
      camera.pitch += 0.2;
      redraw();
    }
    else if(event.keyCode == 65) {
      camera.pitch -= 0.2;
      redraw();
    }
    else if(event.keyCode == 81) {
      camera.yaw += 0.2;
      redraw();
    }
    else if(event.keyCode == 69) {
      camera.yaw -= 0.2;
      redraw();
    }
    else if(event.keyCode == 87) {
      camera.roll += 0.2;
      redraw();
    }
    else if(event.keyCode == 83) {
      camera.roll -= 0.2;
      redraw();
    }
});
 
  loader.load(
    'src/point_cloud.ply',
    async function (geometry) {
      data = new THREE.Points( geometry, material );
      pointsColourData = data.geometry.attributes.pointsColour.array;
      scaleRotationData = data.geometry.attributes.scaleRotation.array;
      ColorData = data.geometry.attributes.Color.array;
      ColorData2 = data.geometry.attributes.Color2.array;
      const scaleRotationSplit = splitList(scaleRotationData,7);
      covarianceMatrices = await computeInverseCovarMatrix(scaleRotationSplit);
      redraw();
    },
    (xhr) => {
        console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
    },
    (error) => {
        console.log(error)
    }

  )

  document.addEventListener('keydown', function(event) {
    if(event.keyCode == 37) {
        console.log(cloud)
    }
  });


  class Camera{
    constructor(x,y,z,pitch,yaw,roll){
      this.x = x;
      this.y = y;
      this.z = z;
      this.pitch = pitch;
      this.yaw = yaw;
      this.roll = roll;
      this.farz = 1000;
      this.closez = 0.1;
      this.hfov = 70;
      this.vfov = 70;
    }

    calcViewMatrix(){
      const yawMat = mat4.create(Math.cos(this.yaw),Math.sin(this.yaw),0,0,-Math.sin(this.yaw),Math.cos(this.yaw),0,0,0,0,1,0,0,0,0,1);
      const rollMat = mat4.create(1,0,0,0,0,Math.cos(this.roll),Math.sin(this.roll),0,0,-Math.sin(this.roll),Math.cos(this.roll),0,0,0,0,1);
      const pitchMat = mat4.create(Math.cos(this.pitch),0,-Math.sin(this.pitch),0,0,1,0,0,Math.sin(this.pitch),0,Math.cos(this.pitch),0,0,0,0,1);
      var RotMat = mat4.multiply(pitchMat,rollMat);
      RotMat = mat4.multiply(yawMat,RotMat);
      const transMat = mat4.create(1,0,0,0,0,1,0,0,0,0,1,0,-this.x,-this.y,-this.z,1)
      return mat4.multiply(RotMat,transMat);
    }
  }

  var time;
  const camera = new Camera(0,-1,-1,1,Math.PI,0);
  var outputs = [];
  var results = [];
  var keyList = [];
  var keyOutputs = [];
  var module;
  var pipeline;
  var bindGroup;
  var encoder;
  var pass;
  var commandBuffer;
  const viewMatrixBuffer = device.createBuffer({
    label: 'uniforms for view matrix',
    size: 16*4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const viewMatrixInverseBuffer = device.createBuffer({
    label: 'uniforms for view matrix',
    size: 16*4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const viewMatrixTransposeInverseBuffer = device.createBuffer({
    label: 'uniforms for view matrix',
    size: 16*4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const offsetBuffer = device.createBuffer({
    label: 'uniforms tile offset',
    size: 2*4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const camVarsBuffer = device.createBuffer({
    label: 'uniforms for quad',
    size: 4*4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const camPosBuffer = device.createBuffer({
    label: 'uniforms for quad',
    size: 3*4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const atom = device.createBuffer({
    label: 'atom buffer',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  
  var workBuffer;
  var InverseCovarBuffer;
  var colorBuffer;
  var colorBuffer2;
  function initBuffers(pointsColourData, covarianceMatrices, colorData, colorData2){
    outputs = [];
    results = [];
    keyList = [];
    keyOutputs = [];
    workBuffer = device.createBuffer({
      label: 'work buffer',
      size: pointsColourData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    //too large, arbitrarily subtracting...
    colorBuffer = device.createBuffer({
      label: 'color Buffer',
      size: colorData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    colorBuffer2 = device.createBuffer({
      label: 'color Buffer',
      size: colorData2.byteLength-40000000,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    for(var i = 0; i<16; i++){
      const outputBuffer = device.createBuffer({
        label: 'output buffer',
        size: (pointsColourData.byteLength/40)*20,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });   
      outputs.push(outputBuffer);
      const resultBuffer = device.createBuffer({
        label: 'result buffer',
        size: (pointsColourData.byteLength/40)*20,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
      results.push(resultBuffer);
      const keysBuffer = device.createBuffer({
        label: 'uniforms for quad',
        size: 4*(pointsColourData.byteLength/40),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      keyList.push(keysBuffer);
      const keysOutputBuffer = device.createBuffer({
        label: 'uniforms for quad',
        size: 4*(pointsColourData.byteLength/40),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      keyOutputs.push(keysOutputBuffer);
    }

    InverseCovarBuffer = device.createBuffer({
      label: 'inv cov buffer',
      size: covarianceMatrices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

  }

  

  async function computeInverseCovarMatrix(scaleRotationSplit){
    
    time = Date.now();
    console.log("setting up calculating inverted covariance matrices")
    module = device.createShaderModule({
      label: 'our hardcoded textured quad shaders',
      code: await (await fetch("src/shaders/covarCompute.txt")).text(),
    });
  
    pipeline = device.createComputePipeline({
      label: 'covar compute pipeline',
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'compute',
      },
    });


    console.log("set up completed in" + (Date.now()-time) + "ms")
    time = Date.now();
    console.log("beginning calculating inverted covariance matrices")

    const workBuffers = [];
    for(var i = 0; i<4; i++){

      const workBuffer = device.createBuffer({
        label: 'work buffer',
        size: (scaleRotationSplit[i].byteLength/7)*9,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const covarDataBuffer = device.createBuffer({
        label: 'covar Data buffer',
        size: scaleRotationSplit[i].byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(covarDataBuffer, 0, scaleRotationSplit[i]);

      const outputBuffer = device.createBuffer({
        label: 'output buffer',
        size: (scaleRotationSplit[i].byteLength/7)*9,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });  



      bindGroup = device.createBindGroup({
        label: 'bindGroup for work buffer',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: covarDataBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
        ],
      });
  
      encoder = device.createCommandEncoder({
        label: 'doubling encoder',
      });
      pass = encoder.beginComputePass({
        label: 'doubling compute pass',
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(100,24,20);
      pass.end();
  
      
      encoder.copyBufferToBuffer(outputBuffer, 0, workBuffer, 0, workBuffer.size);
      commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);

      await workBuffer.mapAsync(GPUMapMode.READ);
      workBuffers.push(new Float32Array(workBuffer.getMappedRange()));
    }

    const l1 = workBuffers[0].length;
    const l2 = workBuffers[1].length;
    const l3 = workBuffers[2].length;
    const l4 = workBuffers[3].length;
    const invcov = new Float32Array(l1+l2+l3+l4);
    invcov.set(workBuffers[0],0);
    invcov.set(workBuffers[1],l1);
    invcov.set(workBuffers[2],l1+l2);
    invcov.set(workBuffers[3],l1+l2+l3);
    
    console.log("calculations completed in" + (Date.now()-time) + "ms")
    return invcov;
  }

  async function compute(covarianceMatrices,pointsColourData,camera,colorData,colorData2,scaleRotationData){
    
    time = Date.now();
    console.log("intizialising determining gaussian tile intersections")

    const covarDataBuffer = device.createBuffer({
      label: 'covar Data buffer',
      size: scaleRotationData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(covarDataBuffer, 0, scaleRotationData);
    //do these modify viewMatrix in place -- dont think so (did print tests for transpose(), did not do for inverse()).
    const viewMatrix = camera.calcViewMatrix();
    const viewMatrixTranspose = mat4.transpose(viewMatrix);
    const viewMatrixInverse = mat4.invert(viewMatrix);
    const viewMatrixTransposeInverse = mat4.invert(viewMatrixTranspose);
    device.queue.writeBuffer(viewMatrixBuffer,0,viewMatrix);
    device.queue.writeBuffer(viewMatrixInverseBuffer,0,viewMatrixInverse);
    device.queue.writeBuffer(viewMatrixTransposeInverseBuffer,0,viewMatrixTransposeInverse);


    module = device.createShaderModule({
      label: 'our hardcoded textured quad shaders',
      code: await (await fetch("src/shaders/computeShader.txt")).text(),
    });
  
    pipeline = device.createComputePipeline({
      label: 'compute pipeline',
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'compute',
      },
    });

    device.queue.writeBuffer(colorBuffer,0,colorData.slice(0,colorData.length/2));
    device.queue.writeBuffer(colorBuffer2,0,colorData2.slice(0,colorData2.length/2));
    device.queue.writeBuffer(workBuffer,0,pointsColourData);
    device.queue.writeBuffer(InverseCovarBuffer,0,covarianceMatrices);



    const offset = new Float32Array(2);
    offset.set([0,0],0);
    device.queue.writeBuffer(offsetBuffer,0,offset);

    const camVars = new Float32Array(4);
    camVars.set([camera.hfov,camera.vfov,camera.farz,camera.closez],0);
    device.queue.writeBuffer(camVarsBuffer,0,camVars);
    const cameraPos = new Float32Array(3);
    cameraPos.set([camera.x,camera.y,camera.z],0);
    device.queue.writeBuffer(camPosBuffer,0,cameraPos);
    const projectionMatrix = mat4.create(Math.atan(camera.hfov/2),0,0,0,0,Math.atan(camera.vfov/2),0,0,0,0,-(camera.farz+camera.closez)/(camera.farz-camera.closez),-1,0,0,-2*(camera.farz*camera.closez)/(camera.farz-camera.closez),0);
    const ProjectionMatrixBuffer = device.createBuffer({
      label: 'uniforms for quad',
      size: 16*4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(ProjectionMatrixBuffer,0,projectionMatrix);

    console.log("init completed in" + (Date.now() - time) + "ms")
    time = Date.now();
    console.log("beginning determining gaussian tile intersections")
    for(var i = 0; i < 4; i++){
      for(var j = 0; j< 4; j++){
        device.queue.writeBuffer(atom,0,new Float32Array([0]));
        offset.set([i,j],0);
        device.queue.writeBuffer(offsetBuffer,0,offset);
        bindGroup = device.createBindGroup({
          label: 'bindGroup for compute shader',
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: workBuffer } },
            { binding: 1, resource: { buffer: viewMatrixBuffer } },
            { binding: 2, resource: { buffer: camVarsBuffer } },
            { binding: 3, resource: { buffer: offsetBuffer } },
            { binding: 4, resource: { buffer: InverseCovarBuffer} },
            { binding: 5, resource: { buffer: atom } },
            { binding: 6, resource: { buffer: outputs[i*4 + j] } },
            { binding: 7, resource: { buffer: keyOutputs[i*4 + j] } },
            { binding: 8, resource: { buffer: ProjectionMatrixBuffer } },
            { binding: 9, resource: { buffer: camPosBuffer } },
            { binding: 10, resource: { buffer: colorBuffer } },
            { binding: 13, resource: { buffer: colorBuffer2 } },
            { binding: 11, resource: { buffer: viewMatrixInverseBuffer } },
            { binding: 12, resource: { buffer: viewMatrixTransposeInverseBuffer } },
            { binding: 14, resource: { buffer: covarDataBuffer } },
          ],
        });

        
        encoder = device.createCommandEncoder({
          label: 'doubling encoder',
        });
        pass = encoder.beginComputePass({
          label: 'doubling compute pass',
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(100,24,20);
        pass.end();

        
        encoder.copyBufferToBuffer(outputs[i*4+j], 0, results[i*4+j], 0, results[i*4+j].size);
        encoder.copyBufferToBuffer(keyOutputs[i*4+j], 0, keyList[i*4+j], 0, keyList[i*4+j].size);
        commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
      }
    }
    console.log("culling completed in" + (Date.now() - time) + "ms")
    time = Date.now();
    console.log("splicing output")
    const finalRes = [];
    const mappedKeys = [];
    var concat;
    for(var i = 0; i < 16; i++){
      await keyList[i].mapAsync(GPUMapMode.READ);
      await results[i].mapAsync(GPUMapMode.READ);
      var res = new Float32Array(results[i].getMappedRange());
      var key = new Int32Array(keyList[i].getMappedRange());
      var lastIndex = res.length - 1;
      while (lastIndex >= 0 && res[lastIndex] === 0) {
          lastIndex--;
      }
      lastIndex++;

      // Remove the trailing zeros using the splice method
      finalRes.push(res.slice(0,lastIndex));
      mappedKeys.push(key.slice(0,lastIndex/20));

    }
    console.log("splicing completed in" + (Date.now() - time) + "ms")
    time = Date.now();
    //let chunkedArray = chunkArray(finalRes[4], 20);
    //console.log(chunkedArray);
    //console.log('input', input);

/*
    console.log("sorting each list with merge sort");
    for(var i=0; i<16; i++){
      let chunkedArray = chunkArray(finalRes[i],20);
      mergeSort(chunkedArray);
    }
    console.log("done sorting each list");
*/

/*
    let concatenatedArray = mappedKeys.reduce((acc, currentArray) => {
      return new Float32Array([...acc, ...currentArray]);
    }, new Float32Array());
    console.log("concatenation completed in " + (Date.now() - time) + "ms")
*/
    time = Date.now();
    console.log("sorting key lists");

    const engine_ctx = new EngineContext();
    await engine_ctx.initialize();
    for( var i = 0; i < 16; i++){
      mappedKeys[i] = await test(mappedKeys[i],engine_ctx);
    }
    console.log("sorting completed in" + (Date.now() - time) + "ms")
/*
    time = Date.now();
    console.log("sorting combined key list")
    let x = await test(concatenatedArray,engine_ctx);
    console.log("sorting completed in" + (Date.now() - time) + "ms")
*/
    //console.log(x)

    render(finalRes,mappedKeys,projectionMatrix);
}





  //the rendering function
  async function render(data,keys,projMat) {

    console.log("rendering")

    const inverseProjMat = mat4.invert(projMat);
    const inverseTransposeProjMat = mat4.invert(mat4.transpose(projMat));
    const ProjectionMatrixInverseBuffer = device.createBuffer({
      label: 'uniforms for quad',
      size: 16*4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const ProjectionMatrixTransposeInverseBuffer = device.createBuffer({
      label: 'uniforms for quad',
      size: 16*4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(ProjectionMatrixInverseBuffer,0,inverseProjMat);
    device.queue.writeBuffer(ProjectionMatrixTransposeInverseBuffer,0,inverseTransposeProjMat);

    const camMat = mat3.create(camera.hfov,0,0,0,camera.vfov,0,0,0,1);
    const ProjectionMatrixBuffer = device.createBuffer({
      label: 'uniforms for quad',
      size: 16*4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const CameraMatrixBuffer = device.createBuffer({
      label: 'uniforms for quad',
      size: 9*8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(ProjectionMatrixBuffer,0,projMat);
    device.queue.writeBuffer(CameraMatrixBuffer,0,camMat);
    for(var i = 0; i < 16; i++){
      const resultBuffer = device.createBuffer({
        label: 'result buffer',
        size: data[i].byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const keyBuffer = device.createBuffer({
        label: 'key buffer',
        size: keys[i].byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
      });
      device.queue.writeBuffer(resultBuffer,0,data[i]);
      device.queue.writeBuffer(keyBuffer,0,keys[i]);
      // Get the current texture from the canvas context and set it as the texture to render to.
      renderPassDescriptor.colorAttachments[0].view =
          context.getCurrentTexture().createView();
  
  
      //create a command encoder and start a render pass.
      const encoder = device.createCommandEncoder({
        label: 'render quad encoder',
      });
      const pass = encoder.beginRenderPass(renderPassDescriptor);
      pass.setPipeline(RenderPipeline);
  
        
      //create a bind group with all the uniform buffers and current cameras texture view and the texture sampler
      var bindGroup = device.createBindGroup({
        layout: RenderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: resultBuffer}},
          { binding: 1, resource: { buffer: CameraMatrixBuffer }},
          { binding: 2, resource: { buffer: ProjectionMatrixBuffer }},
          { binding: 3, resource: { buffer: camPosBuffer } },
          { binding: 4, resource: { buffer: viewMatrixInverseBuffer } },
          { binding: 5, resource: { buffer: viewMatrixTransposeInverseBuffer } },
          { binding: 6, resource: { buffer: ProjectionMatrixInverseBuffer } },
          { binding: 7, resource: { buffer: ProjectionMatrixTransposeInverseBuffer } },
          { binding: 8, resource: { buffer: viewMatrixBuffer } },
  
        ],
      });
  
      //set the bind group and draw.
      pass.setVertexBuffer(0,keyBuffer);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6,keys[i].length);
  
      pass.end();
      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
    }
    

    console.log("done rendering")
  }

  /*
  //create an observer to resize the canvas to match the device
  const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
      const canvas = entry.target;
      const width = entry.contentBoxSize[0].inlineSize;
      const height = entry.contentBoxSize[0].blockSize;
      canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
      canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
      // re-render at new dimensions
      render();
    }
  });
  //use the observer to resize canvas
  observer.observe(canvas);
  */

}


main();