use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interval {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "3m")]
    ThreeMinutes,
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "30m")]
    ThirtyMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "2h")]
    TwoHours,
    #[serde(rename = "4h")]
    FourHours,
    #[serde(rename = "6h")]
    SixHours,
    #[serde(rename = "8h")]
    EightHours,
    #[serde(rename = "12h")]
    TwelveHours,
    #[serde(rename = "1d")]
    OneDay,
    #[serde(rename = "3d")]
    ThreeDays,
    #[serde(rename = "1w")]
    OneWeek,
    #[serde(rename = "1M")]
    OneMonth,
}

impl Interval {
    pub fn as_millis(&self) -> u64 {
        match self {
            Interval::OneMinute => 60_000,
            Interval::ThreeMinutes => 180_000,
            Interval::FiveMinutes => 300_000,
            Interval::FifteenMinutes => 900_000,
            Interval::ThirtyMinutes => 1_800_000,
            Interval::OneHour => 3_600_000,
            Interval::TwoHours => 7_200_000,
            Interval::FourHours => 14_400_000,
            Interval::SixHours => 21_600_000,
            Interval::EightHours => 28_800_000,
            Interval::TwelveHours => 43_200_000,
            Interval::OneDay => 86_400_000,
            Interval::ThreeDays => 259_200_000,
            Interval::OneWeek => 604_800_000,
            Interval::OneMonth => 2_592_000_000,
        }
    }
    
    pub fn as_seconds(&self) -> u64 {
        self.as_millis() / 1000
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::OneMinute => "1m",
            Interval::ThreeMinutes => "3m",
            Interval::FiveMinutes => "5m",
            Interval::FifteenMinutes => "15m",
            Interval::ThirtyMinutes => "30m",
            Interval::OneHour => "1h",
            Interval::TwoHours => "2h",
            Interval::FourHours => "4h",
            Interval::SixHours => "6h",
            Interval::EightHours => "8h",
            Interval::TwelveHours => "12h",
            Interval::OneDay => "1d",
            Interval::ThreeDays => "3d",
            Interval::OneWeek => "1w",
            Interval::OneMonth => "1M",
        }
    }
}
