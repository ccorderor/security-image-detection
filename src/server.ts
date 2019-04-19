require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const sharp = require("sharp");
const fs = require("fs");
const jpeg = require("jpeg-js");
var request = require("request").defaults({ encoding: null });
const NUMBER_OF_CHANNELS = 3;
const cocoSsd = require("@tensorflow-models/coco-ssd");

const uuidv4 = require("uuid/v4");

const express = require("express");
const app = express();

const bodyParser = require("body-parser");
app.use(bodyParser.json()); 
app.use(bodyParser.urlencoded({ extended: true })); 

/**
 * Tensowflow logic
 */

const readImage2 = buf => {
  const pixels = jpeg.decode(buf, true);
  return pixels;
};

const imageByteArray = (image, numChannels) => {
  const pixels = image.data;
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel];
    }
  }

  return values;
};

const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels);
  const outShape = [image.height, image.width, numChannels];
  const input = tf.tensor3d(values, outShape, "int32");

  return input;
};

const classify = async (path, res) => {
  const image = readImage2(path);
  const input = imageToInput(image, NUMBER_OF_CHANNELS);

  const mn_model = await cocoSsd.load();
  //const mn_model = await cocoSsd.load('mobilenet_v2');
  const predictions = await mn_model.detect(input);

  //console.log("classification results:", predictions);

  //const top = aux.split(',');

  // For each bounding box, we generate an SVG rectangle as described here:
  // https://developer.mozilla.org/en-US/docs/Web/SVG/Element/rect
  // Using the top, left, width and height arrays we grabbed earlier

  let boxColor = "#19105f";
  if (predictions.length > 0) {
    let svgRectangles = [];
    for (let i = 0; i < predictions.length; i++) {
      let aux = predictions[i].bbox;
      let svgRectangle = [];
      svgRectangle.push(
        ` <rect height="` +
          aux[3] +
          `" width="` +
          aux[2] +
          `" x="` +
          aux[0] +
          `" y="` +
          aux[1] +
          `"
              style="fill: none; stroke: ` +
          boxColor +
          `; stroke-width: 5"/>`
      );
      let textx = aux[0] + 20;
      let texty = aux[1] + aux[3] + 15;
      svgRectangle.push(
        `<text x="` +
          textx +
          `" y="` +
          texty +
          `" fill="red" text-anchor="middle" alignment-baseline="central">` +
          predictions[i].class +
          `</text>`
      );
      svgRectangles.push(svgRectangle);
    }

    let image2 = sharp(path);
    image2.metadata().then(function(metadata) {
      let svgElement =
        `<svg height="` +
        metadata.height +
        `" width="` +
        metadata.width +
        `" viewbox="0 0 ` +
        metadata.width +
        ` ` +
        metadata.height +
        `" xmlns="http://www.w3.org/2000/svg">`;
      svgElement += svgRectangles.join();
      svgElement += `</svg>`;

      // The SVG string we have crafted above needs to be converted into a Buffer object
      // so that we can use Sharp to overlay it with our image buffer
      const svgElementBuffer = new Buffer(svgElement);

      // Now we create a new image buffer combining the original image buffer with the buffer we generated
      // with our SVG bounding box rectangles

      let uuidFile = uuidv4();
      let outputFile = "tmpimages/" + uuidFile + ".png";

      let imgbuffer = image2
        .overlayWith(svgElementBuffer, { top: 0, left: 0 })
        .toFile(outputFile);

      let response = {
        result: "DETECTED",
        predictions: predictions,
        uuid: uuidFile
      };

      res.send(JSON.stringify(response));
    });
  } else {
    let response = { result: "NOT_DETECTED" };
    res.send(JSON.stringify(response));
  }
};

/**
 * End Tensowflow logic
 */

app.post("/detect", function(req, res) {
  let url = req.body.url;
  //console.log(url);

  request(url, function(err, reqresp, body) {
    classify(body, res).catch(function(e) {
      console.log(e);
      let response = { result: "KO" };
      res.send(JSON.stringify(response));
    });
  });
});

app.get("/image/:uuid", function(req, res) {
  const reqUuid = req.params.uuid;
  const filePath = "tmpimages/" + reqUuid + ".png";
  try {
    const buf = fs.readFileSync(filePath);
  } catch (err) {
    res.status(404);
  }

  res.writeHead(200, { "Content-Type": "image/png" });
  res.end(buf, "binary");
});

app.listen(3000, () => {
  console.log("Server online. Port 3000");
});
