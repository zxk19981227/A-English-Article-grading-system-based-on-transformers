<template>
  <div class="r-echarts-line">
    <div id="main" ref="mychart" style="width: 600px;height:400px;"></div>
  </div>
</template>

<script>
let echarts = require('echarts')
import _ from 'lodash'
export default {
  name: 'bar-report',
  props: {
    data: {
      required: true, // 若是横轴，则此部分为x轴"数值"数据，若为纵轴，则此部分为y轴"数值"数据,此部分数据必传
      type: Object
    },
    title: {
      type: Object | String // 标题，可以只传入标题，其他属性使用默认值，也可自定义title属性，默认无标题
    },
    theme: {
      type: String // dom参数属性，传入theme字符串，theme暂时支持dark和light两种，默认light
    },
    width: String, // dom的宽度，默认600
    height: String, // dom的高度，默认400
    barType: {
      type: String // 柱状图类型/值为xAxis则显示横轴柱状图，值为yAxis则为纵轴柱状图，默认是yAxis
    },
    category: {
      required: true, // 图表分类，如是纵轴，则是x轴的值
      type: Object
    },
    legend: {
      required: true, // 必传，图表上方标识每个颜色柱/线代表什么意思
      type: Array
    },
    dataView: {
      type: Boolean // 是否显示数据视图功能，默认不开启
    },
    magicType: {
      type: Array | String // 是否显示图表之间切换显示功能，默认不开启
    },
    restore: {
      type: Boolean // 是否显示图表还原功能，默认不开启
    },
    saveAsImage: {
      type: Boolean // 是否显示图表图表保存为图片功能，默认不开启
    },
    stack: {
      type: String // 柱状图是否堆叠属性，默认不堆叠
    },
    seriesType: {
      type: String // 控制是绘制柱状图，还是绘制折线图，”bar“:柱状图，”line“:折线图
    }
  },
  data () {
    // 默认title属性
    let baseTitle = {
      text: '', // 主标题
      subtext: '', // 副标题
      left: 'left', // 标题位置：left,center,right
      textStyle: {
        // 文字颜色
        color: '#000000',
        // 字体风格,'normal','italic','oblique'
        fontStyle: 'normal',
        // 字体粗细 'normal','bold','bolder','lighter',100 | 200 | 300 | 400...
        fontWeight: 'bold',
        // 字体系列
        fontFamily: 'sans-serif',
        // 字体大小
        fontSize: 18
      }
    }
    // 如果传入了title的值，则判断处理后重新赋值给baseTitle
    const tempTitle = this.title
    if (tempTitle) {
      debugger
      if (typeof (tempTitle) === 'string') {
        baseTitle.text = tempTitle
      } else if (typeof (tempTitle) === 'object') {
        if (tempTitle.text) {
          baseTitle.text = tempTitle.text
        }
        if (tempTitle.subtext) {
          baseTitle.subtext = tempTitle.subtext
        }
        if (tempTitle.left) {
          baseTitle.left = tempTitle.left
        }
        if (tempTitle.textStyle) {
        // 标题字体颜色
          if (tempTitle.textStyle.color) {
            baseTitle.textStyle.color = tempTitle.textStyle.color
          }
          // 字体风格,'normal','italic','oblique'
          if (tempTitle.textStyle.fontStyle) {
            baseTitle.textStyle.fontStyle = tempTitle.textStyle.fontStyle
          }
          // 字体粗细 'normal','bold','bolder','lighter',100 | 200 | 300 | 400...
          if (tempTitle.textStyle.fontWeight) {
            baseTitle.textStyle.fontWeight = tempTitle.textStyle.fontWeight
          }
          // 字体系列
          if (tempTitle.textStyle.fontFamily) {
            baseTitle.textStyle.fontFamily = tempTitle.textStyle.fontFamily
          }
          // 字体大小
          if (tempTitle.textStyle.fontSize) {
            baseTitle.textStyle.fontSize = tempTitle.textStyle.fontSize
          }
        }
      }
    }
    // 默认dom属性
    let baseDom = {
      theme: 'light',
      renderer: 'canvas',
      opts: {
        width: 600,
        height: 400
      }
    }
    // 判断处理是否传入dom属性值
    if (this.theme) {
      baseDom.theme = this.theme
    }
    if (this.width) {
      baseDom.opts.width = parseInt(this.width)
    }
    if (this.height) {
      baseDom.opts.height = parseInt(this.height)
    }
    let baseType = 'yAxis'
    if (this.barType) {
      baseType = this.barType
    }
    // 默认toolbox值
    var baseToolbox = {
      show: false,
      feature: {
        mark: {show: true},
        dataView: {show: false, readOnly: false}, // 数据视图
        magicType: {show: false, type: []}, // 切换为折线图/柱状图
        restore: {show: false}, // 还原
        saveAsImage: {show: false} // 保存为图片
      }
    }
    // 判断处理传入toolbox属性值
    if (this.dataView) {
      baseToolbox.show = true
      baseToolbox.feature.dataView.show = true
    }
    if (this.magicType) {
      baseToolbox.show = true
      baseToolbox.feature.magicType.show = true
      if (typeof (this.magicType) === 'string') {
        baseToolbox.feature.magicType.type = [this.magicType]
      } else {
        baseToolbox.feature.magicType.type = this.magicType
      }
    }
    if (this.restore) {
      baseToolbox.show = true
      baseToolbox.feature.restore.show = true
    }
    if (this.saveAsImage) {
      baseToolbox.show = true
      baseToolbox.feature.saveAsImage.show = true
    }
    return {
      titleProperty: baseTitle, // title属性值
      domProperty: baseDom, // dom属性值
      type: baseType, // 图表横纵类型
      toolboxProperty: baseToolbox// toolbox属性
    }
  },
  // 页面加载
  mounted () {
    this.setEchart()
  },
  methods: {
    // 绘制图表方法
    setEchart () {
      const _this = this
      let domInit = this.$refs.mychart
      this.myChart = echarts.init(domInit, _this.domProperty.theme, {width: _this.domProperty.opts.width, height: _this.domProperty.opts.height})
      // 指定图表的配置项和数据
      let option = {
        title: {
          text: _this.titleProperty.text,
          subtext: _this.titleProperty.subtext,
          left: _this.titleProperty.left,
          textStyle: _this.titleProperty.textStyle
        },
        tooltip: {
          trigger: 'axis'
        },
        toolbox: this.toolboxProperty,
        legend: {
          data: this.legend
        },
        series: [
        ]
      }
      // 判断传入图表类型是横轴图表还是纵轴图表
      if (this.type === 'xAxis') {
        const tempx = {
          type: 'value'
        }
        option.xAxis = tempx
        option.yAxis = this.category
      } else if (this.type === 'yAxis') {
        const tempy = {
          type: 'value'
        }
        option.yAxis = tempy
        option.xAxis = this.category
      }
      // series数据赋值并判断是否是堆叠图表
      const seriesData = this.data
      if (seriesData) {
        _.forEach(this.legend, function (value) {
          var myBarData = _.get(seriesData, value)
          console.log(value)
          if (_.hasIn(seriesData, value)) {
            var tempSeries = {
              name: value,
              type: _this.seriesType,
              data: myBarData
            }
            if (_this.stack) {
              tempSeries.stack = _this.stack
            }
            option.series.push(tempSeries)
          }
        })
      }
      this.myChart.setOption(option)
    }
  }
}
</script>

<style scoped>

</style>