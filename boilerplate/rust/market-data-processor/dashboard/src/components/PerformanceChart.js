import React from 'react';
import { Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Sample performance data
const performanceData = [
  { date: '2025-05-01', balance: 10000, benchmark: 10000 },
  { date: '2025-05-02', balance: 10120, benchmark: 10050 },
  { date: '2025-05-03', balance: 10200, benchmark: 10070 },
  { date: '2025-05-04', balance: 10180, benchmark: 10090 },
  { date: '2025-05-05', balance: 10250, benchmark: 10110 },
  { date: '2025-05-06', balance: 10310, benchmark: 10140 },
  { date: '2025-05-07', balance: 10275, benchmark: 10125 },
  { date: '2025-05-08', balance: 10340, benchmark: 10145 },
  { date: '2025-05-09', balance: 10420, benchmark: 10170 },
  { date: '2025-05-10', balance: 10390, benchmark: 10160 },
  { date: '2025-05-11', balance: 10450, benchmark: 10190 },
  { date: '2025-05-12', balance: 10520, benchmark: 10210 },
  { date: '2025-05-13', balance: 10480, benchmark: 10200 },
  { date: '2025-05-14', balance: 10550, benchmark: 10230 },
  { date: '2025-05-15', balance: 10600, benchmark: 10260 },
  { date: '2025-05-16', balance: 10680, benchmark: 10290 },
  { date: '2025-05-17', balance: 10650, benchmark: 10280 },
  { date: '2025-05-18', balance: 10720, benchmark: 10310 },
  { date: '2025-05-19', balance: 10790, benchmark: 10340 },
  { date: '2025-05-20', balance: 10850, benchmark: 10370 },
  { date: '2025-05-21', balance: 10920, benchmark: 10400 },
  { date: '2025-05-22', balance: 10880, benchmark: 10390 },
  { date: '2025-05-23', balance: 10950, benchmark: 10420 },
  { date: '2025-05-24', balance: 11020, benchmark: 10450 },
  { date: '2025-05-25', balance: 11090, benchmark: 10480 },
  { date: '2025-05-26', balance: 11150, benchmark: 10510 },
  { date: '2025-05-27', balance: 11220, benchmark: 10540 },
  { date: '2025-05-28', balance: 11180, benchmark: 10530 },
  { date: '2025-05-29', balance: 11250, benchmark: 10560 },
  { date: '2025-05-30', balance: 11320, benchmark: 10590 },
];

function PerformanceChart() {
  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return date.getDate() + '/' + (date.getMonth() + 1);
  };
  
  return (
    <Box sx={{ width: '100%', height: 400 }}>
      <ResponsiveContainer>
        <LineChart
          data={performanceData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tickFormatter={formatXAxis} />
          <YAxis />
          <Tooltip 
            formatter={(value) => [`$${value}`, 'Value']}
            labelFormatter={(label) => new Date(label).toLocaleDateString()}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="balance"
            name="Portfolio"
            stroke="#4caf50"
            activeDot={{ r: 8 }}
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="benchmark"
            name="Benchmark"
            stroke="#9e9e9e"
            strokeWidth={1}
            strokeDasharray="5 5"
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default PerformanceChart;
