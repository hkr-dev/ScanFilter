package com.namangarg.androiddocumentscannerandfilter.Helper;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public final class DocumentImageProcessor {

    private DocumentImageProcessor() {
    }

    public static Mat prepare(Bitmap bitmap) {
        return prepare(bitmap, true, false);
    }

    public static Mat prepare(Bitmap bitmap, boolean geometricCorrection,
                              boolean illuminationCorrection) {
        Mat source = new Mat();
        Utils.bitmapToMat(bitmap, source);
        if (source.channels() == 4) {
            Imgproc.cvtColor(source, source, Imgproc.COLOR_RGBA2BGR);
        } else if (source.channels() == 1) {
            Imgproc.cvtColor(source, source, Imgproc.COLOR_GRAY2BGR);
        }

        Mat corrected = geometricCorrection ? correctPerspective(source) : source;
        if (corrected != source) {
            source.release();
        }
        if (illuminationCorrection) {
            correctIllumination(corrected);
        }
        return corrected;
    }

    public static Mat smoothGrayscale(Bitmap bitmap, float intensity) {
        intensity = clamp(intensity);
        Mat source = prepare(bitmap, true, false);
        Mat lab = new Mat();
        Mat luminance = new Mat();
        Mat background = new Mat();
        Mat rawFloat = new Mat();
        Mat backgroundFloat = new Mat();
        Mat correctedFloat = new Mat();
        Imgproc.cvtColor(source, lab, Imgproc.COLOR_BGR2Lab);
        Core.extractChannel(lab, luminance, 0);
        // A 12% window captures page-scale lighting gradients without washing out strokes.
        int kernel = oddKernel(Math.min(source.cols(), source.rows()), 0.12);
        Imgproc.GaussianBlur(luminance, background, new Size(kernel, kernel), 0);
        luminance.convertTo(rawFloat, CvType.CV_32F);
        background.convertTo(backgroundFloat, CvType.CV_32F);
        Core.add(backgroundFloat, new Scalar(1.0), backgroundFloat);
        Core.multiply(rawFloat, new Scalar(Core.mean(backgroundFloat).val[0]), correctedFloat);
        Core.divide(correctedFloat, backgroundFloat, correctedFloat);
        correctedFloat.convertTo(luminance, CvType.CV_8U);
        Imgproc.createCLAHE(1.5 + intensity * 0.5, new Size(8, 8)).apply(luminance, luminance);
        Mat result = new Mat();
        Mat rawLuminance = extractLuminance(source);
        Core.addWeighted(rawLuminance, 1.0 - intensity, luminance, intensity, 0, result);
        rawLuminance.release();
        release(source, lab, luminance, background, rawFloat, backgroundFloat, correctedFloat);
        return result;
    }

    public static Mat magicColor(Bitmap bitmap, float backgroundWhiteness, float colorBoost) {
        backgroundWhiteness = clamp(backgroundWhiteness);
        colorBoost = clamp(colorBoost);
        Mat source = prepare(bitmap, true, false);
        Mat lab = new Mat();
        Mat light = new Mat();
        Mat background = new Mat();
        Mat lightFloat = new Mat();
        Mat backgroundFloat = new Mat();
        Mat normalized = new Mat();
        Imgproc.cvtColor(source, lab, Imgproc.COLOR_BGR2Lab);
        Core.extractChannel(lab, light, 0);
        // The local white point follows paper shading but remains much larger than text.
        Imgproc.GaussianBlur(light, background,
                new Size(oddKernel(Math.min(source.cols(), source.rows()), 0.10),
                        oddKernel(Math.min(source.cols(), source.rows()), 0.10)), 0);
        light.convertTo(lightFloat, CvType.CV_32F);
        background.convertTo(backgroundFloat, CvType.CV_32F);
        Core.max(backgroundFloat, new Scalar(1), backgroundFloat);
        Core.multiply(lightFloat, new Scalar(255), normalized);
        Core.divide(normalized, backgroundFloat, normalized);
        normalized.convertTo(light, CvType.CV_8U);
        Mat rawLight = new Mat();
        Core.extractChannel(lab, rawLight, 0);
        Core.addWeighted(rawLight, 1.0 - backgroundWhiteness, light, backgroundWhiteness, 0, light);
        Core.insertChannel(light, lab, 0);
        boostLabChannel(lab, 1, colorBoost);
        boostLabChannel(lab, 2, colorBoost);
        Imgproc.cvtColor(lab, source, Imgproc.COLOR_Lab2BGR);
        Mat blur = new Mat();
        Imgproc.GaussianBlur(source, blur, new Size(0, 0), 1.1);
        Core.addWeighted(source, 1.10, blur, -0.10, 0, source);
        release(lab, light, background, lightFloat, backgroundFloat, normalized, rawLight, blur);
        return source;
    }

    public static Mat ecoLighten(Bitmap bitmap, float inkSaveLevel) {
        inkSaveLevel = clamp(inkSaveLevel);
        Mat source = prepare(bitmap, true, false);
        Mat gray = extractLuminance(source);
        Mat grayFloat = new Mat();
        Mat mean = new Mat();
        Mat squaredMean = new Mat();
        Mat variance = new Mat();
        Mat standardDeviation = new Mat();
        Mat threshold = new Mat();
        gray.convertTo(grayFloat, CvType.CV_32F);
        int window = oddKernel(Math.min(source.cols(), source.rows()), 0.035 - inkSaveLevel * 0.015);
        // Mean and squared mean give Sauvola's local standard deviation without Java loops.
        Imgproc.boxFilter(grayFloat, mean, CvType.CV_32F, new Size(window, window));
        Imgproc.sqrBoxFilter(grayFloat, squaredMean, CvType.CV_32F, new Size(window, window));
        Core.multiply(mean, mean, variance);
        Core.subtract(squaredMean, variance, variance);
        Core.max(variance, new Scalar(0), variance);
        Core.sqrt(variance, standardDeviation);
        double k = 0.22 + inkSaveLevel * 0.16;
        Mat contrastTerm = new Mat();
        Core.multiply(standardDeviation, new Scalar(k / 128.0), contrastTerm);
        Core.add(contrastTerm, new Scalar(1.0 - k), contrastTerm);
        Core.multiply(mean, contrastTerm, threshold);
        Mat binary = new Mat();
        Core.compare(grayFloat, threshold, binary, Core.CMP_GT);
        if (inkSaveLevel > 0.05f) {
            Mat kernel = Mat.ones(3, 3, CvType.CV_8U);
            Imgproc.dilate(binary, binary, kernel, new Point(-1, -1), inkSaveLevel > 0.65f ? 2 : 1);
            kernel.release();
        }
        release(source, gray, grayFloat, mean, squaredMean, variance, standardDeviation, threshold, contrastTerm);
        return binary;
    }

    private static Mat extractLuminance(Mat bgr) {
        Mat lab = new Mat();
        Mat luminance = new Mat();
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);
        Core.extractChannel(lab, luminance, 0);
        lab.release();
        return luminance;
    }

    private static void boostLabChannel(Mat lab, int index, float boost) {
        Mat channel = new Mat();
        Mat centered = new Mat();
        Mat magnitude = new Mat();
        Mat denominator = new Mat();
        Core.extractChannel(lab, channel, index);
        channel.convertTo(centered, CvType.CV_32F);
        Core.subtract(centered, new Scalar(128), centered);
        Core.absdiff(centered, new Scalar(0), magnitude);
        Core.multiply(magnitude, new Scalar(0.0025 * boost), denominator);
        Core.add(denominator, new Scalar(1), denominator);
        Core.divide(centered, denominator, centered);
        Core.multiply(centered, new Scalar(1.0 + 0.65 * boost), centered);
        Core.add(centered, new Scalar(128), centered);
        centered.convertTo(channel, CvType.CV_8U);
        Core.insertChannel(channel, lab, index);
        release(channel, centered, magnitude, denominator);
    }

    private static int oddKernel(int dimension, double fraction) {
        int value = Math.max(15, (int) Math.round(dimension * fraction));
        return value % 2 == 0 ? value + 1 : value;
    }

    private static float clamp(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }

    private static void release(Mat... mats) {
        for (Mat mat : mats) mat.release();
    }

    private static Mat correctPerspective(Mat source) {
        Mat gray = new Mat();
        Mat edges = new Mat();
        Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);
        Imgproc.Canny(gray, edges, 60, 180);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges, contours, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        double imageArea = source.rows() * (double) source.cols();
        double largestArea = 0;
        MatOfPoint2f best = null;

        for (MatOfPoint contour : contours) {
            MatOfPoint2f points = new MatOfPoint2f(contour.toArray());
            double area = Math.abs(Imgproc.contourArea(points));
            double perimeter = Imgproc.arcLength(points, true);
            MatOfPoint2f approximation = new MatOfPoint2f();
            Imgproc.approxPolyDP(points, approximation, perimeter * 0.02, true);
            if (approximation.total() == 4 && area > imageArea * 0.70 && area > largestArea) {
                if (best != null) {
                    best.release();
                }
                best = approximation;
                largestArea = area;
            } else {
                approximation.release();
            }
            points.release();
            contour.release();
        }

        Mat result = source;
        if (best != null) {
            Point[] corners = orderCorners(best.toArray());
            double width = Math.max(distance(corners[0], corners[1]), distance(corners[2], corners[3]));
            double height = Math.max(distance(corners[0], corners[3]), distance(corners[1], corners[2]));
            int outputWidth = Math.max(1, (int) Math.round(width));
            int outputHeight = Math.max(1, (int) Math.round(height));
            MatOfPoint2f destination = new MatOfPoint2f(
                    new Point(0, 0), new Point(outputWidth - 1, 0),
                    new Point(outputWidth - 1, outputHeight - 1), new Point(0, outputHeight - 1));
            Mat transform = Imgproc.getPerspectiveTransform(new MatOfPoint2f(corners), destination);
            result = new Mat();
            Imgproc.warpPerspective(source, result, transform,
                    new Size(outputWidth, outputHeight), Imgproc.INTER_CUBIC);
            transform.release();
            destination.release();
            best.release();
        }
        gray.release();
        edges.release();
        return result;
    }

    private static void correctIllumination(Mat image) {
        Mat lab = new Mat();
        Mat light = new Mat();
        Mat background = new Mat();
        Mat lightFloat = new Mat();
        Mat backgroundFloat = new Mat();
        Mat correctedFloat = new Mat();
        Imgproc.cvtColor(image, lab, Imgproc.COLOR_BGR2Lab);
        Core.extractChannel(lab, light, 0);
        int kernel = Math.max(15, Math.min(61, (Math.min(image.cols(), image.rows()) / 8) | 1));
        Imgproc.GaussianBlur(light, background, new Size(kernel, kernel), 0);
        light.convertTo(lightFloat, CvType.CV_32F);
        background.convertTo(backgroundFloat, CvType.CV_32F);
        Mat offset = new Mat(backgroundFloat.size(), CvType.CV_32F, new Scalar(1));
        Core.add(backgroundFloat, offset, backgroundFloat);
        offset.release();
        Core.multiply(lightFloat, new Scalar(Core.mean(backgroundFloat).val[0]), correctedFloat);
        Core.divide(correctedFloat, backgroundFloat, correctedFloat);
        correctedFloat.convertTo(light, CvType.CV_8U);
        Core.insertChannel(light, lab, 0);
        Imgproc.cvtColor(lab, image, Imgproc.COLOR_Lab2BGR);
        lab.release();
        light.release();
        background.release();
        lightFloat.release();
        backgroundFloat.release();
        correctedFloat.release();
    }

    private static Point[] orderCorners(Point[] points) {
        Point[] ordered = new Point[4];
        double minSum = Double.MAX_VALUE;
        double maxSum = -Double.MAX_VALUE;
        double minDifference = Double.MAX_VALUE;
        double maxDifference = -Double.MAX_VALUE;
        for (Point point : points) {
            double sum = point.x + point.y;
            double difference = point.x - point.y;
            if (sum < minSum) { minSum = sum; ordered[0] = point; }
            if (sum > maxSum) { maxSum = sum; ordered[2] = point; }
            if (difference > maxDifference) { maxDifference = difference; ordered[1] = point; }
            if (difference < minDifference) { minDifference = difference; ordered[3] = point; }
        }
        return ordered;
    }

    private static double distance(Point first, Point second) {
        return Math.hypot(first.x - second.x, first.y - second.y);
    }
}