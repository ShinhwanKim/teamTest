package com.example.jsych.teamchicken;

import com.google.gson.JsonObject;

import java.util.ArrayList;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface RetrofitService {

    String url="http://13.124.223.128";

    @GET("/dbparser.php")
    Call<ArrayList<Seat>> getDB(@Query("") String id);


}
