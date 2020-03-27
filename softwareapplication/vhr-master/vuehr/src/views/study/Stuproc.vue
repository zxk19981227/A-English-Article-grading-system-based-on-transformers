<template>
    <div>
        <ve-line :data="chartData"></ve-line>
    </div>
</template>

<script>
    import VeLine from 'v-charts/lib/line.common'
    export default {
        components: { VeLine },
        data () {
            return {
                chartData: {
                    columns: ['date', 'PV'],
                    rows: [
                        { 'date': '02-24', 'PV': 4 },
                        { 'date': '02-25', 'PV': 2 },
                        { 'date': '02-26', 'PV': 5 },
                        { 'date': '02-27', 'PV': 6 },
                        { 'date': '02-28', 'PV': 2},
                        { 'date': '02-29', 'PV': 3 },
                        { 'date': '02-30', 'PV':1},
                        { 'date': '03-01', 'PV': 6 },
                        { 'date': '03-02', 'PV': 4 },
                        { 'date': '03-03', 'PV':0 },
                        { 'date': '03-24', 'PV': 0 }
                    ]
                }
            }
        }
    }
</script>