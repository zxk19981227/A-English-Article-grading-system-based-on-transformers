package org.javaboy.vhr.mapper;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import org.apache.ibatis.annotations.Param;
import org.javaboy.vhr.model.Employee;
import org.javaboy.vhr.model.JobLevel;
import org.javaboy.vhr.model.Map;

import java.util.Date;
import java.util.List;

public interface MapMapper {
     int deleteByPrimaryKey(Integer id);
     int updateByPrimaryKey(Map new_map);
    int insertMap(Map new_map);
    List<Map> getMapByPage(@Param("page") Integer page, @Param("size") Integer size);
    int maxWorkID();
    long getTotal();
}