package com.cxsj.PushDelayDetection.controller;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:8099") // 前端服务器的地址
public class UploadController {

    @GetMapping("/results")
    public List<String> getResults() {
//        String resultsDirectoryPath = new File(System.getProperty("user.dir")).getParent() + "/Datas/results/";
        String resultsDirectoryPath = new File(System.getProperty("user.dir"))+ "/Datas/results/";
        File resultsDirectory = new File(resultsDirectoryPath);
        if (resultsDirectory.exists() && resultsDirectory.isDirectory()) {
            return Arrays.stream(resultsDirectory.listFiles())
                    .filter(file -> !file.isDirectory())
                    .map(File::getName)
                    .collect(Collectors.toList());
        }
        return List.of();
    }

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam("file") MultipartFile file, @RequestParam("model") String model) {
        model="./save_weights/weight_May01_22-47-46.pth";
        try {
            // 获取Spring Boot项目根目录的上一级目录
//            String rootPath = new File(System.getProperty("user.dir")).getParent();
            String rootPath = System.getProperty("user.dir");
            // 定义存储目录
            String storageDirectoryPath = rootPath + "/Datas";
            File storageDirectory = new File(storageDirectoryPath);
            if (!storageDirectory.exists()) {
                storageDirectory.mkdirs();
            }

            // 保存上传的视频文件到指定目录
            File convFile = new File(storageDirectoryPath + "/" + file.getOriginalFilename());
            System.out.println("Saving file to: " + convFile.getAbsolutePath());
            file.transferTo(convFile);

            // 检查文件是否存在
            if (!convFile.exists()) {
                return "File upload failed: File not found after saving.";
            }
            // 打印文件的详细信息
            System.out.println("File name: " + convFile.getName());
            System.out.println("File path: " + convFile.getAbsolutePath());
            System.out.println("File size: " + convFile.length());

            System.out.println("File uploaded successfully: " + convFile.getAbsolutePath());

            // 定义帧图片保存目录
            String framesDirectoryPath = storageDirectoryPath + "/frames/";
            File framesDirectory = new File(framesDirectoryPath);
            if (!framesDirectory.exists()) {
                framesDirectory.mkdirs();
            }

            // 定义结果图片保存目录
            String resultsDirectoryPath = storageDirectoryPath + "/results/";
            File resultsDirectory = new File(resultsDirectoryPath);
            if (!resultsDirectory.exists()) {
                resultsDirectory.mkdirs();
            }

            // 将视频分解为帧并保存到本地
            extractFramesWithJavacv(convFile.getAbsolutePath(), framesDirectoryPath);

            // 调用Python脚本处理所有帧图片
            List<String> results = processFramesWithPython(model, framesDirectoryPath, resultsDirectoryPath);

            // 返回处理结果
            return String.join("\n", results);
        } catch (IOException e) {
            e.printStackTrace();
            return "File upload failed: " + e.getMessage();
        }
    }

    private void extractFramesWithJavacv(String videoFilePath, String framesDirectoryPath) {
        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFilePath)) {
            grabber.start();

            Java2DFrameConverter converter = new Java2DFrameConverter();
            int frameNumber = 0;
            Frame frame;
            while ((frame = grabber.grabImage()) != null) {
                BufferedImage bufferedImage = converter.convert(frame);
                File frameFile = new File(framesDirectoryPath + "frame_" + frameNumber + ".png");
                ImageIO.write(bufferedImage, "png", frameFile);
                frameNumber++;
            }

            grabber.stop();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private List<String> processFramesWithPython(String model, String framesDirectoryPath, String resultsDirectoryPath) {
        List<String> results = new ArrayList<>();
        try {
            // 获取帧图片列表
            List<File> frameFiles = java.nio.file.Files.list(java.nio.file.Paths.get(framesDirectoryPath))
                    .filter(java.nio.file.Files::isRegularFile)
                    .map(path -> path.toFile())
                    .collect(Collectors.toList());

            System.out.println(frameFiles.get(0).getAbsolutePath());
            System.out.println(resultsDirectoryPath);
            for (File frameFile : frameFiles) {
                String outputImagePath = resultsDirectoryPath + frameFile.getName();
                // 使用ProcessBuilder调用Python脚本处理每一帧图片
                ProcessBuilder processBuilder = new ProcessBuilder("python", "YOLOv1_stock/predict_single.py", model, frameFile.getAbsolutePath(), outputImagePath);
                Process process = processBuilder.start();

                // 获取Python脚本的输出
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = reader.readLine()) != null) {
                    result.append(line).append("\n");
                }

                // 等待Python脚本执行完毕
                int exitCode = process.waitFor();
                if (exitCode == 0) {
                    results.add(result.toString());
                } else {
                    results.add("Error processing frame: " + frameFile.getName());
                    System.out.println("Error processing frame: " + frameFile.getName());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            results.add("Error processing frames in directory: " + framesDirectoryPath + " - " + e.getMessage());
        }

        return results;
    }
}
