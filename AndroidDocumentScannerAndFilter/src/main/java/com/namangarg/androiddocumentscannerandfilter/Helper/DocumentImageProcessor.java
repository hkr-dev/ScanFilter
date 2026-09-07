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
        Mat source = new Mat();
        Utils.bitmapToMat(bitmap, source);
        if (source.channels() == 4) {
            Imgproc.cvtColor(source, source, Imgproc.COLOR_RGBA2BGR);
        } else if (source.channels() == 1) {
            Imgproc.cvtColor(source, source, Imgproc.COLOR_GRAY2BGR);
        }

        Mat corrected = correctPerspective(source);
        if (corrected != source) {
            source.release();
        }
        correctIllumination(corrected);
        return corrected;
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
            if (approximation.total() == 4 && area > imageArea * 0.55 && area > largestArea) {
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