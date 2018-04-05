package com.tzutalin.dlibtest;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.LinkedList;
import java.util.List;

public class FaceActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private FaceDet faceDet;
    public LinkedList<Mat> faces;
    private CameraBridgeViewBase mOpenCvCameraView;
    private String TAG = "NTD";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face);
        mOpenCvCameraView = (JavaCameraView)findViewById(R.id.showCamera);
        mOpenCvCameraView.setCvCameraViewListener(this);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        final String targetPath = Constants.getFaceShapeModelPath();
        faces = new LinkedList<>();
        if (!new File(targetPath).exists()) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(FaceActivity.this, "Copy landmark model to " + targetPath, Toast.LENGTH_SHORT).show();
                }
            });
            FileUtils.copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_68_face_landmarks, targetPath);
        }
        faceDet = new FaceDet(targetPath);
        new Thread(new SendImage()).start();

    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }



    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat img1 = inputFrame.rgba();//.submat(120,120+300,230,230 + 500);
        Mat img = new Mat();
        Imgproc.resize(img1,img,new Size(),0.5,0.5,Imgproc.INTER_CUBIC);
        Bitmap bitmap = Bitmap.createBitmap(img.cols(),img.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(img,bitmap);
        List<VisionDetRet> results = faceDet.detect(bitmap);
        //Imgproc.rectangle(inputFrame,new Point(230,120),new Point(230+500,120+300),
        //        new Scalar(255,255,0),2);
        for (final VisionDetRet ret : results) {
            Mat face = img1.submat(ret.getTop()*2,ret.getBottom()*2,ret.getLeft()*2,ret.getRight()*2);
            faces.addLast(face);
            Imgproc.rectangle(img1,new Point(ret.getLeft()*2,ret.getTop()*2),new Point(ret.getRight()*2,ret.getBottom()*2),
                    new Scalar(255,0,0),2);
        }
        return img1;
    }

    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public class SendImage implements Runnable{
        InetAddress serverAddr;
        @Override
        public void run() {
            try {
                serverAddr = InetAddress.getByName("10.11.11.87");
                while (true) {
                    if (!faces.isEmpty()) {
                        try (
                                Socket clientSocket = new Socket(serverAddr, 5000);
                                PrintWriter out = new PrintWriter(clientSocket.getOutputStream())
                        ) {
                            Mat img = faces.removeFirst();
                            Imgproc.cvtColor(img,img,Imgproc.COLOR_BGRA2RGB);
                            MatOfByte mOB = new MatOfByte();
                            Imgcodecs.imencode(".bmp",img,mOB);
                            byte[] imgByte = mOB.toArray();
                            String base64 = Base64.encodeToString(imgByte,Base64.DEFAULT).replaceAll("\n","");
                            out.println(base64.length());
                            out.flush();
                            out.println(base64);
                            out.flush();
                            out.close();
                        } catch (Exception e) {
                            Log.d(TAG, "Error Socket: " + e.getMessage());
                        }
                    }
                }
            }catch (UnknownHostException e){
                Log.d(TAG,"Unknown host: " + e.getMessage());
            }
        }
    }
}
