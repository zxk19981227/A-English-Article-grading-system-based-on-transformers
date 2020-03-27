package org.javaboy.vhr.service;


import org.javaboy.vhr.mapper.MapMapper;
import org.javaboy.vhr.model.Map;
import org.javaboy.vhr.model.RespPageBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
@Service
public class MapService {
    @Autowired
    MapMapper mapMapper;
    @Autowired
    RabbitTemplate rabbitTemplate;
    public final static Logger logger = LoggerFactory.getLogger(EmployeeService.class);
    SimpleDateFormat yearFormat = new SimpleDateFormat("yyyy");
    SimpleDateFormat monthFormat = new SimpleDateFormat("MM");
    DecimalFormat decimalFormat = new DecimalFormat("##.00");
    public Integer addEmp(Map employee) {
        return mapMapper.insertMap(employee);
    }
    public RespPageBean getMapByPage(Integer page, Integer size, Date[] beginDateScope) {
        if (page != null && size != null) {
            page = (page - 1) * size;
        }
        List<Map> data = mapMapper.getMapByPage(page, size);
        Long total = mapMapper.getTotal();
        RespPageBean bean = new RespPageBean();
        bean.setData(data);
        bean.setTotal(total);
        return bean;
    }


    public Integer getMaxMapID() {
        return mapMapper.maxWorkID();
    }

    public Integer deleteEmpByEid(Integer id) {
        return mapMapper.deleteByPrimaryKey(id);
    }

    public Integer updateEmp(Map map) {
        return mapMapper.updateByPrimaryKey(map);
    }

    public Integer addMap(Map map) {
        return mapMapper.insertMap(map);
    }

}
