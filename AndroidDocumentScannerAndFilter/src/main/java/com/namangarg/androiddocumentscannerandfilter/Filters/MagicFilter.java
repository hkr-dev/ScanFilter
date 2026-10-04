package com.namangarg.androiddocumentscannerandfilter.Filters;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;

import com.namangarg.androiddocumentscannerandfilter.Helper.Contrast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MagicFilter {

    public CallBack callBack;

    public interface CallBack<Bitmap>{
        void onComplete(Bitmap bitmap);
    }

    public static void getMagicFilteredImage(final Bitmap bitmap, final MagicFilter.CallBack<Bitmap> callBack){
        Executor executor = Executors.newSingleThreadExecutor();
        final Handler handler = new Handler(Looper.getMainLooper());
        executor.execute(new Runnable() {
            @Override
            public void run() {
                Mat srcArry = new Mat(bitmap.getWidth(),bitmap.getHeight(), CvType.CV_8UC1);
                Utils.bitmapToMat(bitmap, srcArry);
                double contrast_value = Contrast.getContrastLevel(bitmap);

                if (srcArry.channels() >= 3) {
                    Mat channel = new Mat();
                    Imgproc.cvtColor(srcArry, srcArry, Imgproc.COLOR_BGR2HSV);

                    Core.extractChannel(srcArry, channel, 2);
                    CLAHE clahe = Imgproc.createCLAHE();
                    clahe.setClipLimit(1);
                    clahe.apply(channel, channel);
                    Core.insertChannel(channel, srcArry, 2);

                    Core.extractChannel(srcArry, channel, 1);
                    CLAHE clahe2 = Imgproc.createCLAHE();
                    clahe2.setClipLimit(1);
                    clahe2.apply(channel, channel);
                    Core.insertChannel(channel, srcArry, 1);

                    Imgproc.cvtColor(srcArry, srcArry, Imgproc.COLOR_HSV2BGR);
                    channel.release();
                }
                srcArry.convertTo(srcArry, -1,contrast_value,29);
                final Bitmap result = Bitmap.createBitmap(srcArry.cols(),srcArry.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(srcArry, result);
                srcArry.release();

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
