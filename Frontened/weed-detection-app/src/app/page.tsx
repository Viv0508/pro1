"use client"

import type React from "react"
import { useState, useEffect } from "react"
import Image from "next/image"
import Papa from "papaparse"
import { Upload, Loader2, AlertCircle, Leaf, Info } from "lucide-react"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert"

const detectableWeeds = [
  "Carpetweed",
  "CutleafGroundcherry",
  "Eclipta",
  "Goosegrass",
  "MorningGlory",
  "PalmerAmaranth",
  "PricklySida",
  "Purslane",
  "Ragweed",
  "Sicklepod",
  "SpottedSpurge",
  "Waterhemp",
]

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null)
  const [detectedItems, setDetectedItems] = useState<string[]>([])
  const [weedDetails, setWeedDetails] = useState<Record<string, string>>({})
  const [weedGrowthStages, setWeedGrowthStages] = useState<Record<string, string>>({})
  const [roiDetails, setRoiDetails] = useState<
    Record<
      string,
      {
        CostEstimation: string
        ROI_Impact: string
        LocalMarketPriceImpact: string
        LaborCost: string
      }
    >
  >({})
  const [growthPredictions, setGrowthPredictions] = useState<Array<{ predicted_class: number }>>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchWeedDetails()
    fetchGrowthStages()
    fetchRoiDetails()
  }, [])

  const fetchWeedDetails = async () => {
    try {
      const response = await fetch("/weed_management_recommendations_updated.csv")
      const csvText = await response.text()

      const { data } = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
      })

      const details: Record<string, string> = {}
      data.forEach((row: any) => {
        const weedSpecies = row["Weed Species"]
        const recommendations = row["Prevention & Management Recommendations"]
        if (weedSpecies && recommendations) {
          details[weedSpecies.trim()] = recommendations
        }
      })
      setWeedDetails(details)
    } catch (error) {
      console.error("Error fetching CSV data:", error)
      setError("Failed to load weed details. Please try again later.")
    }
  }

  const fetchGrowthStages = async () => {
    try {
      const response = await fetch("/weed_growth_stage.csv")
      const csvText = await response.text()

      const { data } = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
      })

      const stages: Record<string, string> = {}
      data.forEach((row: any) => {
        const label = row["label"]
        const explanation = row["explanation"]
        if (label !== undefined && explanation) {
          stages[label.trim()] = explanation
        }
      })
      setWeedGrowthStages(stages)
    } catch (error) {
      console.error("Error fetching growth stage CSV data:", error)
    }
  }

  const fetchRoiDetails = async () => {
    try {
      const response = await fetch("/weed_roi.csv")
      const csvText = await response.text()

      const { data } = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
      })

      const roiData: Record<
        string,
        {
          CostEstimation: string
          ROI_Impact: string
          LocalMarketPriceImpact: string
          LaborCost: string
        }
      > = {}
      data.forEach((row: any) => {
        const species = row["WeedSpecies"]?.trim()
        if (species) {
          roiData[species] = {
            CostEstimation: row["CostEstimation"],
            ROI_Impact: row["ROI_Impact"],
            LocalMarketPriceImpact: row["LocalMarketPriceImpact"],
            LaborCost: row["LaborCost"],
          }
        }
      })
      setRoiDetails(roiData)
    } catch (error) {
      console.error("Error fetching ROI CSV data:", error)
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setIsLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append("image", selectedFile)

    try {
      const response = await fetch("http://localhost:7000/api/detect", {
        method: "POST",
        body: formData,
      })
      const data = await response.json()

      if (data.annotated_image) {
        setAnnotatedImage(`data:image/jpeg;base64,${data.annotated_image}`)
      }
      setGrowthPredictions(data.cnn_predictions || [])
      setDetectedItems(data.detected_labels || [])
    } catch (error) {
      console.error("Error uploading image:", error)
      setError("Failed to process the image. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  // Group growth predictions by predicted_class
  const groupedGrowthPredictions = growthPredictions.reduce((acc, prediction) => {
    const key = prediction.predicted_class.toString()
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-green-100 dark:from-green-900 dark:to-green-800">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center mb-8 text-green-800 dark:text-green-100 flex items-center justify-center">
          <Leaf className="mr-2 h-8 w-8" />
          Weed Detection System
        </h1>

        <Card className="max-w-2xl mx-auto bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm mb-8">
          <CardHeader>
            <CardTitle className="text-green-700 dark:text-green-300 flex items-center">
              <Info className="mr-2 h-5 w-5" />
              Detectable Weed Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-green-600 dark:text-green-400 mb-2">
              This system can detect the following types of weeds:
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {detectableWeeds.map((weed, index) => (
                <div
                  key={index}
                  className="bg-green-100 dark:bg-green-700/50 p-2 rounded text-sm text-green-800 dark:text-green-200"
                >
                  {weed}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="max-w-2xl mx-auto bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-green-700 dark:text-green-300">Upload an Image</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              <Input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="flex-grow bg-white dark:bg-gray-700"
              />
              <Button
                onClick={handleUpload}
                disabled={!selectedFile || isLoading}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Upload
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive" className="mt-4 max-w-2xl mx-auto">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {annotatedImage && (
          <Card className="mt-8 max-w-2xl mx-auto bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-green-700 dark:text-green-300">Annotated Image</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative aspect-video rounded-lg overflow-hidden">
                <Image src={annotatedImage || "/placeholder.svg"} alt="Annotated" fill className="object-contain" />
              </div>
            </CardContent>
          </Card>
        )}

        {detectedItems.length > 0 && (
          <Card className="mt-8 max-w-2xl mx-auto bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-green-700 dark:text-green-300">Detected Weeds</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-4">
                {detectedItems.map((item, index) => (
                  <li key={index} className="bg-green-100 dark:bg-green-700/50 p-4 rounded-lg">
                    <h3 className="font-semibold text-lg mb-2 text-green-800 dark:text-green-100">{item}</h3>
                    {weedDetails[item] && (
                      <div className="mt-2">
                        <h4 className="text-sm font-semibold text-green-800 dark:text-green-100">
                          Management Recommendations:
                        </h4>
                        <p className="text-sm text-green-700 dark:text-green-200 whitespace-pre-wrap">
                          {weedDetails[item]}
                        </p>
                      </div>
                    )}
                    {roiDetails[item] && (
                      <div className="mt-2 text-sm text-green-700 dark:text-green-200">
                        <p>
                          <strong>Cost Estimation:</strong> {roiDetails[item].CostEstimation}
                        </p>
                        <p>
                          <strong>ROI Impact:</strong> {roiDetails[item].ROI_Impact}
                        </p>
                        <p>
                          <strong>Local Market Price Impact:</strong> {roiDetails[item].LocalMarketPriceImpact}
                        </p>
                        <p>
                          <strong>Labor Cost:</strong> {roiDetails[item].LaborCost}
                        </p>
                      </div>
                    )}
                  </li>
                ))}

              </ul>
            </CardContent>
          </Card>
        )}

        {detectedItems.length > 0 && growthPredictions.length > 0 && (
          <Card className="mt-8 max-w-2xl mx-auto bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-green-700 dark:text-green-300">Growth Stage Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-4">
                {detectedItems.map((weed, index) => {
                  const prediction = growthPredictions[index]
                  const predictedStage =
                    prediction && weedGrowthStages[prediction.predicted_class.toString()]
                      ? weedGrowthStages[prediction.predicted_class.toString()]
                      : "Unknown growth stage"
                  return (
                    <li key={index} className="bg-green-100 dark:bg-green-700/50 p-4 rounded-lg">
                      <div className="flex flex-col">
                        <span className="font-semibold text-lg text-green-800 dark:text-green-100">
                          Weed: {weed}
                        </span>
                        <span className="mt-1 text-sm text-green-700 dark:text-green-200">
                          Growth Stage: {predictedStage}
                        </span>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
