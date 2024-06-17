//package com.cxsj.PushDelayDetection.controller;
//
//import net.sourceforge.tess4j.ITesseract;
//import net.sourceforge.tess4j.Tesseract;
//import net.sourceforge.tess4j.TesseractException;
//import org.bytedeco.javacpp.indexer.FloatIndexer;
//import org.bytedeco.javacv.FFmpegFrameGrabber;
//import org.bytedeco.javacv.Frame;
//import org.bytedeco.javacv.FrameGrabber;
//import org.bytedeco.opencv.global.opencv_dnn;
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_dnn.Net;
//import org.springframework.core.io.ClassPathResource;
//import org.springframework.web.bind.annotation.*;
//import org.springframework.web.multipart.MultipartFile;
//
//import javax.imageio.ImageIO;
//import java.awt.image.BufferedImage;
//import java.io.File;
//import java.io.IOException;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//
//@RestController
//@RequestMapping("/api")
//@CrossOrigin(origins = "http://localhost:8099") // 前端服务器的地址
//public class UploadController2 {
//
//    private static final int CV_8UC3 = 16;
//    private static final int CV_32F = 5;
//
//    @PostMapping("/upload")
//    public String handleFileUpload(@RequestParam("file") MultipartFile file) {
//        try {
//            File convFile = new File(System.getProperty("java.io.tmpdir") + "/" + file.getOriginalFilename());
//            file.transferTo(convFile);
//            String result = detectAndRecognize(convFile);
//            return "File processed successfully: " + result;
//        } catch (IOException e) {
//            return "File upload failed: " + e.getMessage();
//        }
//    }
//
//    private String detectAndRecognize(File videoFile) {
//        try (FrameGrabber grabber = new FFmpegFrameGrabber(videoFile)) {
//            grabber.start();
//
//            Frame frame;
//            ITesseract tesseract = new Tesseract();
//            StringBuilder ocrResults = new StringBuilder();
//
//            // 读取YOLOv5模型
//            ClassPathResource resource = new ClassPathResource("models/yolov5s.onnx");
//            Net net = opencv_dnn.readNetFromONNX(resource.getFile().getAbsolutePath());
//
//            while ((frame = grabber.grab()) != null) {
//                // 将Frame转换为BufferedImage
//                BufferedImage bufferedImage = new BufferedImage(frame.imageWidth, frame.imageHeight, BufferedImage.TYPE_3BYTE_BGR);
//                bufferedImage.getRaster().setDataElements(0, 0, frame.imageWidth, frame.imageHeight, frame.image[0].array());
//
//                // 将BufferedImage转换为Mat
//                Mat image = bufferedImageToMat(bufferedImage);
//
//                // 进行目标检测
//                Mat blob = opencv_dnn.blobFromImage(image, 1.0 / 255.0, new Size(640, 640), new Scalar(0.0), true, false, CV_32F);
//                net.setInput(blob);
//                Mat detections = net.forward();
//
//                // 解析检测结果并进行OCR
//                FloatIndexer indexer = detections.createIndexer();
//                for (int i = 0; i < detections.size(2); i++) {
//                    float[] data = new float[7];
//                    for (int j = 0; j < 7; j++) {
//                        data[j] = indexer.get(0, 0, i, j);
//                    }
//                    int classId = (int) data[1];
//                    float confidence = data[2];
//                    int x = (int) (data[3] * image.cols());
//                    int y = (int) (data[4] * image.rows());
//                    int width = (int) (data[5] * image.cols() - x);
//                    int height = (int) (data[6] * image.rows() - y);
//
//                    if (confidence > 0.5) {
//                        Mat region = new Mat(image, new Rect(x, y, width, height));
//                        BufferedImage regionImage = matToBufferedImage(region);
//                        String ocrResult = tesseract.doOCR(regionImage);
//                        ocrResults.append(ocrResult).append("\n");
//                    }
//                }
//            }
//
//            grabber.stop();
//            return ocrResults.toString();
//        } catch (IOException e) {
//            e.printStackTrace();
//            return "Error during processing: " + e.getMessage();
////        } catch (FrameGrabber.Exception e) {
////            e.printStackTrace();
////            return "Error during processing: " + e.getMessage();
//        } catch (TesseractException e) {
//            e.printStackTrace();
//            return "Error during OCR processing: " + e.getMessage();
//        } catch (Exception e) {
//            e.printStackTrace();
//            return "Error during processing: " + e.getMessage();
//        }
//    }
//
//    private Mat bufferedImageToMat(BufferedImage bufferedImage) throws IOException {
//        byte[] pixels = ((java.awt.image.DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
//        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CV_8UC3);
//        mat.data().put(pixels);
//        return mat;
//    }
//
//    private BufferedImage matToBufferedImage(Mat mat) throws IOException {
//        int type = BufferedImage.TYPE_BYTE_GRAY;
//        if (mat.channels() > 1) {
//            type = BufferedImage.TYPE_3BYTE_BGR;
//        }
//        byte[] b = new byte[mat.channels() * mat.cols() * mat.rows()];
//        mat.data().get(b);
//        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
//        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), b);
//        return image;
//    }
//}
