"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Progress } from "@/components/ui/progress"
import {
  GraduationCap,
  TrendingUp,
  BookOpen,
  Users,
  AlertTriangle,
  CheckCircle,
  Brain,
  Target,
  BarChart3,
  Lightbulb,
  Clock,
  BookMarked,
  Wifi,
  Home,
  DollarSign,
  User,
} from "lucide-react"

interface PredictionResult {
  predicted_grade: string
  confidence: number
  grade_probabilities: Record<string, number>
  recommendations: string[]
  score_breakdown: {
    academic_score: number
    study_impact: number
    resource_impact: number
    socio_impact: number
    total_score: number
    percentage: number
  }
  feature_analysis: {
    study_efficiency: string
    academic_trend: string
    resource_utilization: string
  }
  pass_status: "PASS" | "FAIL"
}

export default function StudentGradePrediction() {
  const [formData, setFormData] = useState({
    age: "",
    gender: "",
    family_income: "",
    parent_education: "",
    employment_status: "",
    study_hours_per_week: "",
    library_visits_per_month: "",
    online_resource_usage: "",
    ct1_score: "",
    ct2_score: "",
    assignment_score: "",
    presentation_score: "",
    midterm_score: "",
    final_score: "",
  })

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handlePredict = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      if (response.ok) {
        const result = await response.json()
        setPrediction(result)
      } else {
        console.error("Prediction failed")
      }
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setLoading(false)
    }
  }

  const getGradeColor = (grade: string) => {
    const gradeColors: Record<string, string> = {
      "A+": "from-emerald-500 to-green-600",
      A: "from-emerald-400 to-green-500",
      "A-": "from-green-400 to-emerald-500",
      "B+": "from-blue-500 to-indigo-600",
      B: "from-blue-400 to-blue-500",
      "B-": "from-sky-400 to-blue-500",
      "C+": "from-amber-400 to-yellow-500",
      C: "from-yellow-400 to-amber-500",
      "C-": "from-orange-400 to-amber-500",
      D: "from-red-400 to-orange-500",
      F: "from-red-500 to-red-600",
    }
    return gradeColors[grade] || "from-gray-400 to-gray-500"
  }

  const getGradeBadgeColor = (grade: string) => {
    const gradeColors: Record<string, string> = {
      "A+": "bg-emerald-500 text-white border-emerald-600",
      A: "bg-emerald-400 text-white border-emerald-500",
      "A-": "bg-green-400 text-white border-green-500",
      "B+": "bg-blue-500 text-white border-blue-600",
      B: "bg-blue-400 text-white border-blue-500",
      "B-": "bg-sky-400 text-white border-sky-500",
      "C+": "bg-amber-400 text-white border-amber-500",
      C: "bg-yellow-400 text-gray-800 border-yellow-500",
      "C-": "bg-orange-400 text-white border-orange-500",
      D: "bg-red-400 text-white border-red-500",
      F: "bg-red-500 text-white border-red-600",
    }
    return gradeColors[grade] || "bg-gray-400 text-white border-gray-500"
  }

  const getPassStatusColor = (status: string) => {
    return status === "PASS" ? "text-emerald-600" : "text-red-600"
  }

  const getPassStatusIcon = (status: string) => {
    return status === "PASS" ? <CheckCircle className="h-6 w-6" /> : <AlertTriangle className="h-6 w-6" />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="text-center">
            <div className="flex items-center justify-center gap-3 mb-3">
              <div className="p-2 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl">
                <GraduationCap className="h-8 w-8 text-white" />
              </div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                Student Grade Prediction System
              </h1>
            </div>
            <p className="text-slate-600 max-w-2xl mx-auto text-lg">
              Advanced AI-powered grade prediction with personalized insights.
              <span className="font-semibold text-indigo-600"> Pass mark: 40%</span>
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="space-y-6">
            <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-t-lg">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <BookOpen className="h-6 w-6" />
                  Student Information
                </CardTitle>
                <CardDescription className="text-indigo-100">
                  Enter comprehensive student details for accurate grade prediction
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6 space-y-8">
                {/* Demographics */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <User className="h-5 w-5 text-indigo-600" />
                    <h3 className="font-semibold text-slate-800 text-lg">Demographics</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="age" className="text-slate-700 font-medium">
                        Age
                      </Label>
                      <Input
                        id="age"
                        type="number"
                        placeholder="20"
                        className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                        value={formData.age}
                        onChange={(e) => handleInputChange("age", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="gender" className="text-slate-700 font-medium">
                        Gender
                      </Label>
                      <Select value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                        <SelectTrigger className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500">
                          <SelectValue placeholder="Select gender" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="Male">Male</SelectItem>
                          <SelectItem value="Female">Female</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                <Separator className="bg-slate-200" />

                {/* Family Background */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Home className="h-5 w-5 text-purple-600" />
                    <h3 className="font-semibold text-slate-800 text-lg">Family Background</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="family_income" className="text-slate-700 font-medium flex items-center gap-2">
                        <DollarSign className="h-4 w-4" />
                        Family Income (BDT)
                      </Label>
                      <Input
                        id="family_income"
                        type="number"
                        placeholder="500,000"
                        className="border-slate-300 focus:border-purple-500 focus:ring-purple-500"
                        value={formData.family_income}
                        onChange={(e) => handleInputChange("family_income", e.target.value)}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="parent_education" className="text-slate-700 font-medium">
                          Parent Education
                        </Label>
                        <Select
                          value={formData.parent_education}
                          onValueChange={(value) => handleInputChange("parent_education", value)}
                        >
                          <SelectTrigger className="border-slate-300 focus:border-purple-500 focus:ring-purple-500">
                            <SelectValue placeholder="Select education" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="High School">High School</SelectItem>
                            <SelectItem value="College">College</SelectItem>
                            <SelectItem value="Bachelor">Bachelor</SelectItem>
                            <SelectItem value="Master">Master</SelectItem>
                            <SelectItem value="PhD">PhD</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="employment_status" className="text-slate-700 font-medium">
                          Employment Status
                        </Label>
                        <Select
                          value={formData.employment_status}
                          onValueChange={(value) => handleInputChange("employment_status", value)}
                        >
                          <SelectTrigger className="border-slate-300 focus:border-purple-500 focus:ring-purple-500">
                            <SelectValue placeholder="Select status" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Unemployed">Unemployed</SelectItem>
                            <SelectItem value="Part-time">Part-time</SelectItem>
                            <SelectItem value="Full-time">Full-time</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </div>

                <Separator className="bg-slate-200" />

                {/* Study Habits */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Clock className="h-5 w-5 text-emerald-600" />
                    <h3 className="font-semibold text-slate-800 text-lg">Study Habits</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="study_hours" className="text-slate-700 font-medium">
                        Study Hours/Week
                      </Label>
                      <Input
                        id="study_hours"
                        type="number"
                        step="0.1"
                        placeholder="15.5"
                        className="border-slate-300 focus:border-emerald-500 focus:ring-emerald-500"
                        value={formData.study_hours_per_week}
                        onChange={(e) => handleInputChange("study_hours_per_week", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="library_visits" className="text-slate-700 font-medium flex items-center gap-2">
                        <BookMarked className="h-4 w-4" />
                        Library Visits/Month
                      </Label>
                      <Input
                        id="library_visits"
                        type="number"
                        placeholder="8"
                        className="border-slate-300 focus:border-emerald-500 focus:ring-emerald-500"
                        value={formData.library_visits_per_month}
                        onChange={(e) => handleInputChange("library_visits_per_month", e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="online_usage" className="text-slate-700 font-medium flex items-center gap-2">
                      <Wifi className="h-4 w-4" />
                      Online Resource Usage
                    </Label>
                    <Select
                      value={formData.online_resource_usage}
                      onValueChange={(value) => handleInputChange("online_resource_usage", value)}
                    >
                      <SelectTrigger className="border-slate-300 focus:border-emerald-500 focus:ring-emerald-500">
                        <SelectValue placeholder="Select usage frequency" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Never">Never</SelectItem>
                        <SelectItem value="Rarely">Rarely</SelectItem>
                        <SelectItem value="Sometimes">Sometimes</SelectItem>
                        <SelectItem value="Often">Often</SelectItem>
                        <SelectItem value="Always">Always</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Separator className="bg-slate-200" />

                {/* Academic Scores */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-4">
                    <BarChart3 className="h-5 w-5 text-blue-600" />
                    <h3 className="font-semibold text-slate-800 text-lg">Academic Scores</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="ct1_score" className="text-slate-700 font-medium">
                        CT1 Score <span className="text-slate-500">(out of 10)</span>
                      </Label>
                      <Input
                        id="ct1_score"
                        type="number"
                        placeholder="8"
                        max="10"
                        className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                        value={formData.ct1_score}
                        onChange={(e) => handleInputChange("ct1_score", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="ct2_score" className="text-slate-700 font-medium">
                        CT2 Score <span className="text-slate-500">(out of 10)</span>
                      </Label>
                      <Input
                        id="ct2_score"
                        type="number"
                        placeholder="9"
                        max="10"
                        className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                        value={formData.ct2_score}
                        onChange={(e) => handleInputChange("ct2_score", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="assignment_score" className="text-slate-700 font-medium">
                        Assignment <span className="text-slate-500">(out of 20)</span>
                      </Label>
                      <Input
                        id="assignment_score"
                        type="number"
                        placeholder="15"
                        max="20"
                        className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                        value={formData.assignment_score}
                        onChange={(e) => handleInputChange("assignment_score", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="presentation_score" className="text-slate-700 font-medium">
                        Presentation <span className="text-slate-500">(out of 15)</span>
                      </Label>
                      <Input
                        id="presentation_score"
                        type="number"
                        placeholder="12"
                        max="15"
                        className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                        value={formData.presentation_score}
                        onChange={(e) => handleInputChange("presentation_score", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="midterm_score" className="text-slate-700 font-medium">
                        Midterm <span className="text-slate-500">(out of 30)</span>
                      </Label>
                      <Input
                        id="midterm_score"
                        type="number"
                        placeholder="25"
                        max="30"
                        className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                        value={formData.midterm_score}
                        onChange={(e) => handleInputChange("midterm_score", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="final_score" className="text-slate-700 font-medium">
                        Final Exam <span className="text-slate-500">(out of 40)</span>
                      </Label>
                      <Input
                        id="final_score"
                        type="number"
                        placeholder="35"
                        max="40"
                        className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                        value={formData.final_score}
                        onChange={(e) => handleInputChange("final_score", e.target.value)}
                      />
                    </div>
                  </div>
                </div>

                <Button
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-200"
                >
                  {loading ? (
                    <div className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Analyzing...
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      Predict Grade
                    </div>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="space-y-6">
            <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-t-lg">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <TrendingUp className="h-6 w-6" />
                  Prediction Results
                </CardTitle>
                <CardDescription className="text-emerald-100">
                  AI-powered analysis with personalized insights
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                {prediction ? (
                  <div className="space-y-6">
                    {/* Pass/Fail Status */}
                    <div
                      className={`p-6 rounded-xl border-2 ${
                        prediction.pass_status === "PASS"
                          ? "bg-gradient-to-r from-emerald-50 to-green-50 border-emerald-200"
                          : "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
                      }`}
                    >
                      <div
                        className={`flex items-center justify-center gap-3 mb-3 ${getPassStatusColor(prediction.pass_status)}`}
                      >
                        {getPassStatusIcon(prediction.pass_status)}
                        <h3 className="text-2xl font-bold">{prediction.pass_status}</h3>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-slate-800 mb-1">
                          {prediction.score_breakdown.percentage}%
                        </div>
                        <p className="text-slate-600">
                          {prediction.pass_status === "PASS" ? "Above 40% pass mark ✓" : "Below 40% pass mark ⚠️"}
                        </p>
                      </div>
                    </div>

                    {/* Main Prediction */}
                    <div className="text-center p-8 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
                      <div className="mb-6">
                        <h3 className="text-xl font-semibold text-slate-700 mb-4">Predicted Grade</h3>
                        <div
                          className={`inline-flex items-center justify-center w-24 h-24 rounded-2xl text-white text-3xl font-bold bg-gradient-to-br ${getGradeColor(prediction.predicted_grade)} shadow-lg`}
                        >
                          {prediction.predicted_grade}
                        </div>
                      </div>
                      <div className="flex items-center justify-center gap-2 text-slate-600">
                        <Target className="h-5 w-5" />
                        <span className="text-lg">
                          Confidence:{" "}
                          <span className="font-semibold text-slate-800">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </span>
                      </div>
                    </div>

                    {/* Grade Probabilities */}
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800 text-lg flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-indigo-600" />
                        Grade Probabilities
                      </h4>
                      <div className="space-y-3">
                        {Object.entries(prediction.grade_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 5)
                          .map(([grade, prob]) => (
                            <div key={grade} className="flex items-center gap-4">
                              <Badge className={`${getGradeBadgeColor(grade)} min-w-[3rem] justify-center font-bold`}>
                                {grade}
                              </Badge>
                              {(grade === "D" || grade === "F") && (
                                <span className="text-xs text-red-600 font-medium">(Below pass mark)</span>
                              )}
                              <div className="flex-1">
                                <Progress value={prob * 100} className="h-3 bg-slate-200" />
                              </div>
                              <span className="text-sm font-semibold text-slate-700 min-w-[3rem] text-right">
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>

                    {/* Enhanced Score Breakdown */}
                    <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                      <h4 className="font-semibold text-blue-800 mb-4 text-lg flex items-center gap-2">
                        <BarChart3 className="h-5 w-5" />
                        Detailed Score Analysis
                      </h4>
                      <div className="grid grid-cols-2 gap-6 text-sm">
                        <div className="text-center p-4 bg-white/60 rounded-lg">
                          <div className="text-3xl font-bold text-blue-600 mb-1">
                            {prediction.score_breakdown.academic_score}%
                          </div>
                          <div className="text-blue-700 font-medium">Academic (60%)</div>
                        </div>
                        <div className="text-center p-4 bg-white/60 rounded-lg">
                          <div className="text-3xl font-bold text-emerald-600 mb-1">
                            {prediction.score_breakdown.study_impact}%
                          </div>
                          <div className="text-emerald-700 font-medium">Study Habits (20%)</div>
                        </div>
                        <div className="text-center p-4 bg-white/60 rounded-lg">
                          <div className="text-3xl font-bold text-purple-600 mb-1">
                            {prediction.score_breakdown.resource_impact}%
                          </div>
                          <div className="text-purple-700 font-medium">Resources (15%)</div>
                        </div>
                        <div className="text-center p-4 bg-white/60 rounded-lg">
                          <div className="text-3xl font-bold text-amber-600 mb-1">
                            {prediction.score_breakdown.socio_impact}%
                          </div>
                          <div className="text-amber-700 font-medium">Background (5%)</div>
                        </div>
                      </div>
                      <div className="mt-6 pt-6 border-t border-blue-200">
                        <div className="text-center p-4 bg-white/80 rounded-lg">
                          <div className="text-4xl font-bold text-indigo-600 mb-1">
                            {prediction.score_breakdown.percentage}%
                          </div>
                          <div className="text-indigo-700 font-semibold text-lg">
                            Final Percentage {prediction.score_breakdown.percentage >= 40 ? "(PASS)" : "(FAIL)"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Feature Analysis */}
                    <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200">
                      <h4 className="font-semibold text-green-800 mb-4 text-lg flex items-center gap-2">
                        <Brain className="h-5 w-5" />
                        Performance Analysis
                      </h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-white/60 rounded-lg">
                          <span className="text-green-700 font-medium">Study Efficiency:</span>
                          <Badge
                            variant={prediction.feature_analysis.study_efficiency === "High" ? "default" : "secondary"}
                            className="font-semibold"
                          >
                            {prediction.feature_analysis.study_efficiency}
                          </Badge>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-white/60 rounded-lg">
                          <span className="text-green-700 font-medium">Academic Trend:</span>
                          <Badge
                            variant={
                              prediction.feature_analysis.academic_trend === "Improving"
                                ? "default"
                                : prediction.feature_analysis.academic_trend === "Declining"
                                  ? "destructive"
                                  : "secondary"
                            }
                            className="font-semibold"
                          >
                            {prediction.feature_analysis.academic_trend}
                          </Badge>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-white/60 rounded-lg">
                          <span className="text-green-700 font-medium">Resource Use:</span>
                          <Badge
                            variant={
                              prediction.feature_analysis.resource_utilization === "Excellent" ? "default" : "secondary"
                            }
                            className="font-semibold"
                          >
                            {prediction.feature_analysis.resource_utilization}
                          </Badge>
                        </div>
                      </div>
                    </div>

                    {/* Personalized Recommendations */}
                    <div
                      className={`p-6 rounded-xl border-2 ${
                        prediction.pass_status === "FAIL"
                          ? "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
                          : prediction.score_breakdown.percentage < 50
                            ? "bg-gradient-to-r from-amber-50 to-yellow-50 border-amber-200"
                            : "bg-gradient-to-r from-emerald-50 to-green-50 border-emerald-200"
                      }`}
                    >
                      <h4
                        className={`font-semibold mb-4 text-lg flex items-center gap-2 ${
                          prediction.pass_status === "FAIL"
                            ? "text-red-800"
                            : prediction.score_breakdown.percentage < 50
                              ? "text-amber-800"
                              : "text-emerald-800"
                        }`}
                      >
                        <Lightbulb className="h-5 w-5" />
                        {prediction.pass_status === "FAIL"
                          ? "Critical Action Required"
                          : "Personalized Recommendations"}
                      </h4>
                      <div className="space-y-3">
                        {prediction.recommendations.map((rec, index) => (
                          <div
                            key={index}
                            className={`flex items-start gap-3 p-3 rounded-lg bg-white/60 ${
                              prediction.pass_status === "FAIL"
                                ? "text-red-700"
                                : prediction.score_breakdown.percentage < 50
                                  ? "text-amber-700"
                                  : "text-emerald-700"
                            }`}
                          >
                            <span
                              className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${
                                prediction.pass_status === "FAIL"
                                  ? "bg-red-500"
                                  : prediction.score_breakdown.percentage < 50
                                    ? "bg-amber-500"
                                    : "bg-emerald-500"
                              }`}
                            />
                            <span className="font-medium">{rec}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-16 text-slate-500">
                    <div className="mb-6">
                      <Users className="h-16 w-16 mx-auto mb-4 opacity-30" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2 text-slate-600">Ready for Analysis</h3>
                    <p className="text-lg">
                      Fill in the student information and click "Predict Grade" to see detailed results
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-slate-200">
            <p className="text-slate-600 text-lg">
              Powered by advanced machine learning algorithms with 40% minimum pass mark standard.
            </p>
            <div className="flex items-center justify-center gap-6 mt-4 text-sm text-slate-500">
              <span className="flex items-center gap-1">
                <Brain className="h-4 w-4" />
                Random Forest ML
              </span>
              <span className="flex items-center gap-1">
                <Target className="h-4 w-4" />
                85%+ Accuracy
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                Real-time Analysis
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
