package org.javaboy.vhr.service;


import org.javaboy.vhr.mapper.MapMapper;
import org.javaboy.vhr.mapper.RegMapper;
import org.javaboy.vhr.model.Map;
import org.javaboy.vhr.model.Reg;
import org.javaboy.vhr.model.RespPageBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class ReService {
    @Autowired
    RegMapper mapMapper;
    @Autowired
    RabbitTemplate rabbitTemplate;
    public final static Logger logger = LoggerFactory.getLogger(EmployeeService.class);
    SimpleDateFormat yearFormat = new SimpleDateFormat("yyyy");
    SimpleDateFormat monthFormat = new SimpleDateFormat("MM");
    DecimalFormat decimalFormat = new DecimalFormat("##.00");


    public Integer addMap(Reg map) {
        BCryptPasswordEncoder encoder=new BCryptPasswordEncoder();
        map.setPassword(encoder.encode(map.getPassword()));
        return mapMapper.insert(map);
    }
}
