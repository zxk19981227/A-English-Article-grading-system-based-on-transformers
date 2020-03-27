package org.javaboy.vhr.model;

import java.io.Serializable;

public class Mind implements Serializable {
    private int id;
    private String con_name;
    private String content;
    public void setId(int id){
        this.id=id;
    }
    public int getId(){
        return id;
    }
    public void setCon_name(String name){
        this.con_name=name;
    }
    public String getCon_name(){
        return this.con_name;
    }
    public String getContent(){
        return content;
    }
    public void setContent(String detail){
        this.content=detail;
    }
}


