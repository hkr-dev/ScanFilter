package com.namangarg.androiddocumentscannerandfilter.Filters;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;

import com.namangarg.androiddocumentscannerandfilter.Helper.DocumentImageProcessor;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class LightenFilter {

    public CallBack callBack;

    public interface CallBack<Bitmap>{
        void onComplete(Bitmap bitmap);
    }

    public static void getLightenFilteredImage(final Bitmap bitmap, final LightenFilter.CallBack<Bitmap> callBack){
        Executor executor = Executors.newSingleThreadExecutor();
        final Handler handler = new Handler(Looper.getMainLooper());
        executor.execute(new Runnable() {
            @Override
            public void run() {
                Mat mat = DocumentImageProcessor.prepare(bitmap);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2Lab);
                java.util.List<Mat> channels = new java.util.ArrayList<>();
                org.opencv.core.Core.split(mat, channels);
                channels.get(0).convertTo(channels.get(0), -1, 1.08, 8);
                org.opencv.core.Core.merge(channels, mat);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_Lab2BGR);
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
