package com.namangarg.androiddocumentscannerandfilter.Filters;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;

import com.namangarg.androiddocumentscannerandfilter.Helper.DocumentImageProcessor;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class BlackAndWhiteFilter{

    public CallBack callBack;

    public interface CallBack<Bitmap>{
        void onComplete(Bitmap bitmap);
    }

    public static void getBlackAndWhiteFilteredImage(final Bitmap bitmap, final CallBack<Bitmap> callBack){
        // adaptive threshold with median blur
        // works best on text written on blank paper instead of ruled ones.

        Executor executor = Executors.newSingleThreadExecutor();
        final Handler handler = new Handler(Looper.getMainLooper());
        executor.execute(new Runnable() {
            @Override
            public void run() {
                Mat mat = DocumentImageProcessor.prepare(bitmap);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                Imgproc.GaussianBlur(mat, mat, new org.opencv.core.Size(3, 3), 0);
                Imgproc.adaptiveThreshold(mat, mat,255,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,31,7);
                final Bitmap result = Bitmap.createBitmap(mat.cols(),mat.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mat, result);
                mat.release();

                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        callBack.onComplete(result);
                    }
                });
            }
        });
    }
}
