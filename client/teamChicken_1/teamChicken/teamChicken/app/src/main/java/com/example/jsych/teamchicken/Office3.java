package com.example.jsych.teamchicken;

import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.util.ArrayList;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class Office3 extends AppCompatActivity {

    TextView[] seat;
    RetrofitService http;
    boolean status,check;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_office3);
        http = new Retrofit.Builder().baseUrl(RetrofitService.url).addConverterFactory(GsonConverterFactory.create()).build().create(RetrofitService.class);
        seat=new TextView[12];
        seat[0]=findViewById(R.id.seat1);
        seat[1]=findViewById(R.id.seat2);
        seat[2]=findViewById(R.id.seat3);
        seat[3]=findViewById(R.id.seat4);
        seat[4]=findViewById(R.id.seat5);
        seat[5]=findViewById(R.id.seat6);
        seat[6]=findViewById(R.id.seat7);
        seat[7]=findViewById(R.id.seat8);
        seat[8]=findViewById(R.id.seat9);
        seat[9]=findViewById(R.id.seat10);
        seat[10]=findViewById(R.id.seat11);
        seat[11]=findViewById(R.id.seat12);
        for(int i=0;i<seat.length;i++){
            seat[i].setBackgroundColor(Color.parseColor("#ff99cc00"));
        }


        status=true;
        check=true;
        Thread thread=new Thread(new Runnable() {
            @Override
            public void run() {
                while (true){
                    Log.d("ㅂㅂㅂ","true");
                    while (status){
                        Log.d("ㅂㅂㅂ","status");
                        while (check){
                            Log.d("ㅂㅂㅂ","check");
                            check=false;
                            http.getDB("").enqueue(new Callback<ArrayList<Seat>>() {
                                @Override
                                public void onResponse(Call<ArrayList<Seat>> call, Response<ArrayList<Seat>> response) {
                                    ArrayList<Seat> seats=response.body();
                                    for(int i=0;i<seats.size();i++){
                                        Log.d("ㅂㅂㅂ",seats.get(i).getNo()+"");
                                        Log.d("ㅂㅂㅂ",seats.get(i).getCheck()+"");
                                        if(seats.get(i).getCheck()==1){
                                            seat[i].setBackgroundColor(Color.parseColor("#ffff4444"));
                                        }else{
                                            seat[i].setBackgroundColor(Color.parseColor("#ff99cc00"));

                                        }
                                    }
                                    check=true;

                                }

                                @Override
                                public void onFailure(Call<ArrayList<Seat>> call, Throwable t) {

                                }
                            });
                        }
                    }
                }
            }
        });
        thread.start();


    }

    @Override
    protected void onResume() {
        super.onResume();
        status=true;


    }

    @Override
    protected void onPause() {
        super.onPause();
        status=false;
    }
}
