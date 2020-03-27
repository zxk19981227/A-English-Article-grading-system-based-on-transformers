package org.javaboy.vhr.model;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

public class Reg {
    private String username;
    private String password;

    public void setUsername(String na) {
        this.username = na;
    }
    public void setPassword(String pas){
     this.password=pas;
    }
    public String getPassword(){
        return password;
    }
    public String getUsername(){
        return username;
    }
}
