package org.javaboy.vhr.reg;


import org.apache.ibatis.annotations.Delete;
import org.javaboy.vhr.model.*;
import org.javaboy.vhr.service.*;
import org.javaboy.vhr.utils.POIUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.stereotype.Service;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Date;
import java.util.List;


@RestController
@RequestMapping("/doregister")
public class RegController{
    @Autowired
    ReService mapService;

    @PostMapping("/")
    public RespBean addEmp(@RequestBody Reg employee) {

        if (mapService.addMap(employee) == 1) {
            return RespBean.ok("注册成功!");
        }
        return RespBean.error("添加失败!");
    }

}
