//extentions : allow user to upload images to use
import {vec4,mat4} from 'https://webgpufundamentals.org/3rdparty/wgpu-matrix.module.js';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import { mergeSort } from './merge.js';
  
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
  canvas.width = 1000;
  canvas.height = 400;
  const context = canvas.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
  });
  /*
  //Load shader from shaders/shader.txt
  const module = device.createShaderModule({
    label: 'our hardcoded textured quad shaders',
    code: await (await fetch("src/shaders/shader.txt")).text(),
  });


  //initialize the rendering pipeline, with alpha blending.
  const pipeline = device.createRenderPipeline({
    label: 'hardcoded textured quad pipeline',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs',
    },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [{ format: presentationFormat, blend: {color: {srcFactor : "src-alpha", dstFactor: "dst-alpha", operation:"add"}, alpha: {srcFactor : "src-alpha", dstFactor: "dst-alpha", operation : "add"}} }],
    },
  });
*/
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
    pointsColour: ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity'],
    scaleRotation: ['scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'],
    //17 elements
    //f_dc TAKES VALUES > 1? IS THIS THE COLOUR CHANNEL?? OPACITY ALSO TAKES NEGATIVE VALUES AND VALUES > 1, MAYBE FINE DUE TO THE VALUES RETURNED BY GAUSSIAN FUNCTION.
  })

  var data;
 
  loader.load(
    'src/point_cloud.ply',
    async function (geometry) {
      data = new THREE.Points( geometry, material );
      var pointsColourData = data.geometry.attributes.pointsColour.array;
      var scaleRotationData = data.geometry.attributes.scaleRotation.array;
      const scaleRotationSplit = splitList(scaleRotationData,7);
      const covarianceMatrices = await computeInverseCovarMatrix(scaleRotationSplit);
      initBuffers(pointsColourData,covarianceMatrices);
      //console.log(points);
      //console.log("heheheheh");
      compute(covarianceMatrices,pointsColourData,camera);
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
      this.hfov = 90;
      this.vfov = 90;
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


  const camera = new Camera(0,2,0,0,0,0);
  var outputs = [];
  var results = [];
  var atomsList = [];
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
  const atom = device.createBuffer({
    label: 'atom buffer',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  
  var workBuffer;
  var InverseCovarBuffer;
  function initBuffers(pointsColourData, covarianceMatrices){
    workBuffer = device.createBuffer({
      label: 'work buffer',
      size: pointsColourData.byteLength,
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
      const atomsBuffer = device.createBuffer({
        label: 'uniforms for quad',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      atomsList.push(atomsBuffer);
    }

    InverseCovarBuffer = device.createBuffer({
      label: 'inv cov buffer',
      size: covarianceMatrices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

  }

  

  async function computeInverseCovarMatrix(scaleRotationSplit){
    
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
    console.log("done calculating inverted covariance matrices");
    return invcov;
  }

  async function compute(covarianceMatrices,pointsColourData,camera){
    
    
    console.log("intizialising determining gaussian tile intersections")
    
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

    device.queue.writeBuffer(workBuffer,0,pointsColourData);




    device.queue.writeBuffer(InverseCovarBuffer,0,covarianceMatrices);



    const offset = new Float32Array(2);
    offset.set([0,0],0);
    device.queue.writeBuffer(offsetBuffer,0,offset);

    const camVars = new Float32Array(4);
    camVars.set([camera.hfov,camera.vfov,camera.farz,camera.closez],0);
    device.queue.writeBuffer(camVarsBuffer,0,camVars);


    

    console.log("beginning determining gaussian tile intersections")
    for(var i = 0; i < 4; i++){
      for(var j = 0; j< 4; j++){
        device.queue.writeBuffer(atom,0,new Float32Array([0]));
        offset.set([i,j],0);
        device.queue.writeBuffer(offsetBuffer,0,offset);
        bindGroup = device.createBindGroup({
          label: 'bindGroup for work buffer',
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: workBuffer } },
            { binding: 1, resource: { buffer: viewMatrixBuffer } },
            { binding: 2, resource: { buffer: camVarsBuffer } },
            { binding: 3, resource: { buffer: offsetBuffer } },
            { binding: 4, resource: { buffer: InverseCovarBuffer} },
            { binding: 5, resource: { buffer: atom } },
            { binding: 6, resource: { buffer: outputs[i*4 + j] } },
            { binding: 11, resource: { buffer: viewMatrixInverseBuffer } },
            { binding: 12, resource: { buffer: viewMatrixTransposeInverseBuffer } },
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
        encoder.copyBufferToBuffer(atom, 0, atomsList[i*4+j], 0, atomsList[i*4+j].size);
        commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
      }
    }

    console.log("splicing output")
    const finalRes = [];
    for(var i = 0; i < 16; i++){
      await atomsList[i].mapAsync(GPUMapMode.READ);
      await results[i].mapAsync(GPUMapMode.READ);
      var res = new Float32Array(results[i].getMappedRange());
      var lastIndex = res.length - 1;
      while (lastIndex >= 0 && res[lastIndex] === 0) {
          lastIndex--;
      }
      
      // Remove the trailing zeros using the splice method
      finalRes.push(res.slice(0,lastIndex + 1),0);
    }
    console.log("done splicing output")

    console.log(new Float32Array(atomsList[2].getMappedRange()));
    console.log("done determining gaussian tile intersections")
    let chunkedArray = chunkArray(finalRes[4], 20);
    console.log(chunkedArray);
    //console.log('input', input);

/*
    console.log("sorting each list with merge sort");
    for(var i=0; i<16; i++){
      let chunkedArray = chunkArray(finalRes[i],20);
      mergeSort(chunkedArray);
    }
    console.log("done sorting each list");
*/

}





  //the rendering function
  function render() {


    // Get the current texture from the canvas context and set it as the texture to render to.
    renderPassDescriptor.colorAttachments[0].view =
        context.getCurrentTexture().createView();


    //create a command encoder and start a render pass.
    const encoder = device.createCommandEncoder({
      label: 'render quad encoder',
    });
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);

      
    //create a bind group with all the uniform buffers and current cameras texture view and the texture sampler
    var bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        
      ],
    });

    //set the bind group and draw.
    //pass.setBindGroup(0, bindGroup);
    pass.draw(6);

    pass.end();
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
  }

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


}


main();