package org.javaboy.vhr.controller;

import org.javaboy.vhr.model.Reg;
import org.javaboy.vhr.model.RespBean;
import org.javaboy.vhr.service.ReService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @作者 江南一点雨
 * @公众号 江南一点雨
 * @微信号 a_java_boy
 * @GitHub https://github.com/lenve
 * @博客 http://wangsong.blog.csdn.net
 * @网站 http://www.javaboy.org
 * @时间 2019-09-21 21:15
 */
@RestController
public class LoginController {
    @Autowired
    ReService mapService;
    @GetMapping("/login")
    public RespBean login() {
        return RespBean.error("尚未登录，请登录!");
    }

}
