package org.javaboy.vhr.model;

import java.io.Serializable;

public class Map implements Serializable {
    private int id;
    private String map_name;
    private double altitude;
    private double longitude;
    private String detail;
    public void setId(int id){
        this.id=id;
    }
    public int getId(){
        return id;
    }
    public void setMap_name(String name){
        this.map_name=name;
    }
    public String getMap_name(){
        return this.map_name;
    }
    public void setAltitude(double altitude){
        this.altitude=altitude;
    }
    public double getAltitude(){
        return this.altitude;
    }
    public void setLongitude(double longitude){
        this.longitude=longitude;
    }
    public double getLongitude(){
        return this.longitude;
    }
    public String getDetail(){
        return detail;
    }
    public void setDetail(String detail){
        this.detail=detail;
    }
}
