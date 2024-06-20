package com.cxsj.PushDelayDetection.controller;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
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
        String resultsDirectoryPath = new File(System.getProperty("user.dir")) + "/Datas/results/";
        File resultsDirectory = new File(resultsDirectoryPath);
        if (resultsDirectory.exists() && resultsDirectory.isDirectory()) {
            return Arrays.stream(resultsDirectory.listFiles())
                    .filter(file -> !file.isDirectory())
                    .map(File::getName)
                    .sorted((f1, f2) -> {
                        int n1 = Integer.parseInt(f1.replaceAll("\\D", ""));
                        int n2 = Integer.parseInt(f2.replaceAll("\\D", ""));
                        return Integer.compare(n1, n2);
                    })
                    .collect(Collectors.toList());
        }
        return List.of();
    }

    @GetMapping("/results/{fileName}")
    public ResponseEntity<InputStreamResource> getResultImage(@PathVariable String fileName) {
        String resultsDirectoryPath = new File(System.getProperty("user.dir")) + "/Datas/results/";
        File file = new File(resultsDirectoryPath + fileName);
        if (file.exists() && file.isFile()) {
            try {
                FileInputStream fileInputStream = new FileInputStream(file);
                InputStreamResource resource = new InputStreamResource(fileInputStream);

                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.IMAGE_PNG);

                return new ResponseEntity<>(resource, headers, HttpStatus.OK);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam("file") MultipartFile file, @RequestParam("model") String model) {
//        model = "save_weights/weight_Jun16_09-31-26.pth";
        model = "C:\\Users\\Simon\\GithubProjects\\PushDelayDetectionDemo\\YOLOv1_stock\\save_weights\\weight_Jun16_09-31-26.pth";
        try {
            // 获取Spring Boot项目根目录的上一级目录
            String rootPath = System.getProperty("user.dir");
            // 定义存储目录
            String storageDirectoryPath = rootPath + "/Datas";
            File storageDirectory = new File(storageDirectoryPath);
            if (!storageDirectory.exists()) {
                storageDirectory.mkdirs();
            }

            // 清空Datas目录
            clearDirectory(storageDirectory);

            // 保存上传的视频文件到指定目录
            File convFile = new File(storageDirectoryPath + "/" + file.getOriginalFilename());
            System.out.println("Saving file to: " + convFile.getAbsolutePath());
            file.transferTo(convFile);

            // 检查文件是否存在
            if (!convFile.exists()) {
                return "File upload failed: File not found after saving.";
            }

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
            String result = processFramesWithPython(model, framesDirectoryPath, resultsDirectoryPath);

            // 返回处理结果
            return result;
        } catch (IOException e) {
            e.printStackTrace();
            return "File upload failed: " + e.getMessage();
        }
    }

    private void clearDirectory(File directory) {
        if (directory.exists() && directory.isDirectory()) {
            for (File file : directory.listFiles()) {
                if (file.isDirectory()) {
                    clearDirectory(file);
                }
                file.delete();
            }
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

    private String processFramesWithPython(String model, String framesDirectoryPath, String resultsDirectoryPath) {
        try {
            String condaEnvName = "yolov1_py365";
            ProcessBuilder processBuilder = new ProcessBuilder("conda", "run", "-n", condaEnvName, "python", "YOLOv1_stock/predict_single.py", model, framesDirectoryPath, resultsDirectoryPath);
            Process process = processBuilder.start();

            BufferedReader stdOutput = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));

            StringBuilder result = new StringBuilder();
            String line;

            // 读取标准输出
            System.out.println("Standard Output:");
            while ((line = stdOutput.readLine()) != null) {
                result.append(line).append("\n");
                System.out.println(line);
            }

            // 读取错误输出
            System.out.println("Error Output:");
            StringBuilder errorOutput = new StringBuilder();
            while ((line = stdError.readLine()) != null) {
                errorOutput.append(line).append("\n");
                System.err.println(line);
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return result.toString();
            } else {
                System.out.println("Error processing frames.");
                System.err.println("Error Output:\n" + errorOutput.toString());
                return "Error processing frames.";
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Error processing frames in directory: " + framesDirectoryPath + " - " + e.getMessage();
        }
    }
}
