var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
require("@tensorflow/tfjs-node");
var tf = require("@tensorflow/tfjs");
var sharp = require("sharp");
var fs = require("fs");
var jpeg = require("jpeg-js");
var request = require("request").defaults({ encoding: null });
var NUMBER_OF_CHANNELS = 3;
var cocoSsd = require("@tensorflow-models/coco-ssd");
var uuidv4 = require("uuid/v4");
var express = require("express");
var app = express();
var path = require('path');
var bodyParser = require("body-parser");
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
/**
 * Tensowflow logic
 */
var readImage2 = function (buf) {
    var pixels = jpeg.decode(buf, true);
    return pixels;
};
var imageByteArray = function (image, numChannels) {
    var pixels = image.data;
    var numPixels = image.width * image.height;
    var values = new Int32Array(numPixels * numChannels);
    for (var i = 0; i < numPixels; i++) {
        for (var channel = 0; channel < numChannels; ++channel) {
            values[i * numChannels + channel] = pixels[i * 4 + channel];
        }
    }
    return values;
};
var imageToInput = function (image, numChannels) {
    var values = imageByteArray(image, numChannels);
    var outShape = [image.height, image.width, numChannels];
    var input = tf.tensor3d(values, outShape, "int32");
    return input;
};
var classify = function (path, res) { return __awaiter(_this, void 0, void 0, function () {
    var image, input, mn_model, predictions, boxColor, svgRectangles_1, i, aux, svgRectangle, textx, texty, image2_1, response;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                image = readImage2(path);
                input = imageToInput(image, NUMBER_OF_CHANNELS);
                return [4 /*yield*/, cocoSsd.load()];
            case 1:
                mn_model = _a.sent();
                return [4 /*yield*/, mn_model.detect(input)];
            case 2:
                predictions = _a.sent();
                boxColor = "#19105f";
                if (predictions.length > 0) {
                    svgRectangles_1 = [];
                    for (i = 0; i < predictions.length; i++) {
                        aux = predictions[i].bbox;
                        svgRectangle = [];
                        svgRectangle.push(" <rect height=\"" +
                            aux[3] +
                            "\" width=\"" +
                            aux[2] +
                            "\" x=\"" +
                            aux[0] +
                            "\" y=\"" +
                            aux[1] +
                            "\"\n              style=\"fill: none; stroke: " +
                            boxColor +
                            "; stroke-width: 5\"/>");
                        textx = aux[0] + 20;
                        texty = aux[1] + aux[3] + 15;
                        svgRectangle.push("<text x=\"" +
                            textx +
                            "\" y=\"" +
                            texty +
                            "\" fill=\"red\" text-anchor=\"middle\" alignment-baseline=\"central\">" +
                            predictions[i].class +
                            "</text>");
                        svgRectangles_1.push(svgRectangle);
                    }
                    image2_1 = sharp(path);
                    image2_1.metadata().then(function (metadata) {
                        var svgElement = "<svg height=\"" +
                            metadata.height +
                            "\" width=\"" +
                            metadata.width +
                            "\" viewbox=\"0 0 " +
                            metadata.width +
                            " " +
                            metadata.height +
                            "\" xmlns=\"http://www.w3.org/2000/svg\">";
                        svgElement += svgRectangles_1.join();
                        svgElement += "</svg>";
                        // The SVG string we have crafted above needs to be converted into a Buffer object
                        // so that we can use Sharp to overlay it with our image buffer
                        var svgElementBuffer = new Buffer(svgElement);
                        // Now we create a new image buffer combining the original image buffer with the buffer we generated
                        // with our SVG bounding box rectangles
                        var uuidFile = uuidv4();
                        var outputFile = "tmpimages/" + uuidFile + ".png";
                        var imgbuffer = image2_1
                            .overlayWith(svgElementBuffer, { top: 0, left: 0 })
                            .toFile(outputFile);
                        var response = {
                            result: "DETECTED",
                            predictions: predictions,
                            uuid: uuidFile
                        };
                        res.send(JSON.stringify(response));
                    });
                }
                else {
                    response = { result: "NOT_DETECTED" };
                    res.send(JSON.stringify(response));
                }
                return [2 /*return*/];
        }
    });
}); };
/**
 * End Tensowflow logic
 */
app.post("/detect", function (req, res) {
    var url = req.body.url;
    //console.log(url);
    request(url, function (err, reqresp, body) {
        classify(body, res).catch(function (e) {
            console.log(e);
            var response = { result: "KO" };
            res.send(JSON.stringify(response));
        });
    });
});
var public = path.join(__dirname, '../tmpimages');
app.use('/tmpimages/', express.static(public));
app.listen(3000, function () {
    console.log("Server online. Port 3000");
});
