package org.javaboy.vhr.mapper;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import org.apache.ibatis.annotations.Param;
import org.javaboy.vhr.model.Employee;
import org.javaboy.vhr.model.JobLevel;
import org.javaboy.vhr.model.Map;
import org.javaboy.vhr.model.Mind;

import java.util.Date;
import java.util.List;

public interface MindMapper {
    int deleteByPrimaryKey(Integer id);
    int updateByPrimaryKey(Mind new_map);
    int insertMap(Mind new_map);
    List<Map> getMapByPage(@Param("page") Integer page, @Param("size") Integer size);
    int maxWorkID();
    long getTotal();
}