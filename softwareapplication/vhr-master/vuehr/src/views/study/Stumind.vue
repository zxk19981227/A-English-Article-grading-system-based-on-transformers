<template>
    <div>
        <div>
            <div style="display: flex;justify-content: space-between">

                <div>
                    <el-button type="primary" icon="el-icon-plus" @click="showAddEmpView">
                        添加心得
                    </el-button>

                </div>
            </div>
            <el-table :data="maps" border stripe size="mini">
                <el-table-column type="selection" align="left" width="55"></el-table-column>
                <el-table-column prop="id" label="编号" fixed width="55" align="left"></el-table-column>
                <el-table-column prop="con_name" label="标题" width="200" align="left"></el-table-column>
                <el-table-column prop="content" label="详情" width="700" align="left"></el-table-column>

                <el-table-column label="操作" align="center">
                    <template slot-scope="scope">
                    <el-button @click="showEditEmpView(scope.row)" style="padding: 3px" size="mini">编辑</el-button>
                        <el-button  @click="deletemap(scope.row)" style="padding: 3px" size="mini">
                            删除心得
                        </el-button>
                    </template>

                </el-table-column>

            </el-table>
            <div class="box" style=" display:flex,justify-content:flex-end">
                <div id="container" style="width:1500px; height:600px"></div>
            </div>
            <div style="display: flex;justify-content: flex-end,width:40%; height:40%">
                <el-pagination
                        background
                        @size-change="sizeChange"
                        @current-change="currentChange"
                        layout="sizes, prev, pager, next, jumper, ->, total, slot"
                        :total="total">
                </el-pagination>
            </div>


        </div>
            <el-dialog
                  :title="title"
                  :visible.sync="dialogVisible"
                  width="80%">
                    <div>
                        <el-form :model="emp" :rules="rules" ref="empForm">
                            <el-row>
                                <el-col :span="6">
                                    <el-form-item label="名称:" prop="con_name">
                                        <el-input size="mini"   style="width: 150px" prefix-icon="el-icon-edit" v-model="emp.con_name"
                                                  placeholder="请输入心得名称"></el-input>
                                    </el-form-item>
                                </el-col>

                            </el-row>
                            <el-row>
                                <el-col :span="6">
                                    <el-form-item label="内容:" prop="content">
                                       <el-input   :autosize="{ minRows: 2, maxRows: 4}" size="mini" style="width: 150px" prefix-icon="el-icon-edit" v-model="emp.content"
                                            placeholder="请输入心得内容"></el-input>
                                    </el-form-item>
                                </el-col>
                            </el-row>


                        </el-form>
                    </div>
                    <span slot="footer" class="dialog-footer">
            <el-button @click="dialogVisible = false">取 消</el-button>
            <el-button type="primary" @click="doAddEmp">确 定</el-button>
          </span>
          </el-dialog>
    </div>
</template>

<script>
    export default {
        name: "Mind",
        data() {
            return {
                minds: [],
                total: 0,
                title:'',
                dialogVisible:false,
                currentPage: 1,
                currentSize: 10,
                currentSalary: null,
                emp:{
                    con_name:"啥也不知道",
                    content:"宇宙中心"
                }
            }
        },
        mounted() {
            this.initEmps();
        },
        methods: {
            showEditEmpView(data){
                this.title = '编辑地图信息';
                this.emp = data;
                this.dialogVisible = true;
            },
            init (data) {
                /*let map = new AMap.Map('container', {
                    resizeEnable: true,
                    rotateEnable:true,
                    pitchEnable:true,
                    zoom: 17,
                    pitch:80,
                    rotation:-15,
                    viewMode:'3D',//开启3D视图,默认为关闭
                    buildingAnimation:true,//楼块出现是否带动画

                    expandZoomRange:true,
                    zooms:[3,20],
                    center:[data.longitude,data.altitude]
                })*/
                //window.location.href ='http://localhost:8080/map.html?lat='+data.altitude+'&lon='+data.longitude+'&zoom=17.4&tilt=0.0&rotation=0.0'
                //this.$router.push({path: 'http://localhost:8080/map.html?lat='+data.latitude+'&lon='+data.longitude+'&zoom=17.4&tilt=0.0&rotation=0.0'})
            },
            deletemap(data){
                this.$confirm('此操作将永久删除【' + data.id + '】, 是否继续?', '提示', {
                    confirmButtonText: '确定',
                    cancelButtonText: '取消',
                    type: 'warning'
                }).then(() => {
                    this.deleteRequest("/map/" + data.id).then(resp => {
                        if (resp) {
                            this.initEmps();
                        }
                    })
                }).catch(() => {
                    this.$message({
                        type: 'info',
                        message: '已取消删除'
                    });
                });
            },

            doAddEmp() {
                if (this.emp.id) {
                    this.$refs['empForm'].validate(valid => {
                        if (valid) {
                            this.putRequest("/study/mind/", this.emp).then(resp => {
                                if (resp) {
                                    this.dialogVisible = false;
                                    this.initEmps();
                                }
                            })
                        }
                    });
                } else {
                    this.$refs['empForm'].validate(valid => {
                        if (valid) {
                            this.postRequest("/study/mind/", this.emp).then(resp => {
                                if (resp) {
                                    this.dialogVisible = false;
                                    this.initEmps();
                                }
                            })
                        }
                    });
                }
            },

            sizeChange(size) {
                this.currentSize = size;
                this.initEmps();
            },
            currentChange(page) {
                this.currentPage = page;
                this.initEmps();
            },

            showAddEmpView() {
                this.emptyEmp();
                this.title = '添加地图';
                this.getMaxWordID();
                this.dialogVisible = true;
            },
            getMaxWordID() {
                this.getRequest("/study/mind/maxWorkID").then(resp => {
                    if (resp) {
                        this.emp.workID = resp.obj;
                    }
                })
            },
            emptyEmp(){
            this.emp={
            id:"",
            con_name:"",
            content:"",

              }
            },
            showPop(data) {
                if (data) {
                    this.currentSalary = data.id;
                } else {
                    this.currentSalary = null;
                }
            },

            initEmps() {
                this.getRequest("/study/mind/?page=" + this.currentPage + '&size=' + this.currentSize).then(resp => {
                    if (resp) {
                        this.maps = resp.data;
                        this.total = resp.total;
                    }
                })
            }
        }
    }
</script>
<style scoped>
.map-container {
  position: relative;
  left: 0;
  top: 0;
  width: 10%;
  height: 10%;
}
</style>
