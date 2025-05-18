import React, { useState, useEffect } from "react";
// import "./App.css";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { ProgressBar, Spinner } from "react-bootstrap";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [preprocessingOption, setPreprocessingOption] =
    useState("noPreprocessing");
  const [denoisingOption, setDenoisingOption] = useState("Disable");
  const [inputSpectrum, setInputSpectrum] = useState(null);
  const [preprocessedSpectrum, setPreprocessedSpectrum] = useState(null);
  const [denoisedSpectrum, setDenoisedSpectrum] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentOperation, setCurrentOperation] = useState(""); // Added state for current operation
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const [showClassification, setShowClassification] = useState(false);

  // Individual loading states
  const [loadingInput, setLoadingInput] = useState(false);
  const [loadingPreprocess, setLoadingPreprocess] = useState(false);
  const [loadingDenoise, setLoadingDenoise] = useState(false);
  const [loadingClassify, setLoadingClassify] = useState(false);

  // Chart configurations
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Wavenumber (cm⁻¹)" },
      },
      y: { title: { display: true, text: "Intensity" } },
    },
  };

  const chartOptionsWithLegend = {
    ...chartOptions,
    plugins: {
      legend: { display: true },
    },
  };

  const inputChartData = {
    labels: inputSpectrum?.wavenumbers || [],
    datasets: [
      {
        label: "Input Spectrum",
        data: inputSpectrum?.intensities || [],
        borderColor: "rgb(6, 0, 90)",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  };

  const preprocessedChartData = {
    labels: preprocessedSpectrum?.wavenumbers || [],
    datasets: [
      {
        label: "Preprocessed Spectrum",
        data: preprocessedSpectrum?.intensities || [],
        borderColor: "rgb(0, 71, 128)",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  };

  const denoisedChartData = {
    labels: denoisedSpectrum?.wavenumbers || [],
    datasets: [
      {
        label: "Denoised Spectrum",
        data: denoisedSpectrum?.intensities || [],
        borderColor: "rgb(117, 2, 2)",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  };

  const comparisonChartData = {
    labels: denoisedSpectrum?.wavenumbers || [],
    datasets: [
      {
        label: "Reference Spectrum",
        data: classificationResult?.clean_spectrum || [],
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 2,
        pointRadius: 0,
      },
      {
        label: "Denoised Spectrum",
        data: denoisedSpectrum?.intensities || [],
        borderColor: "rgb(117, 2, 2)",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (
      selectedFile.type !== "text/csv" &&
      !selectedFile.name.endsWith(".csv")
    ) {
      alert("Please upload a valid CSV file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const fileContent = event.target.result.replace(/^\uFEFF/, "");
      const rows = fileContent
        .trim()
        .split(/\r?\n/)
        .filter((row) => row.trim() !== "");

      if (rows.length === 0) {
        alert("The CSV file is empty.");
        return;
      }

      const firstRow = rows[0];
      const delimiter = firstRow.includes(";")
        ? ";"
        : firstRow.includes("\t")
        ? "\t"
        : ",";

      if (!rows.every((row) => row.split(delimiter).length === 2)) {
        alert("CSV must have exactly 2 columns: wavenumbers and intensity.");
        return;
      }

      setFile(selectedFile);
    };
    reader.readAsText(selectedFile);
  };

  const handleClear = () => {
    if (!window.confirm("Are you sure you want to clear all data?")) return;

    setFile(null);
    setInputSpectrum(null);
    setPreprocessedSpectrum(null);
    setDenoisedSpectrum(null);
    setClassificationResult(null);
    setError(null);
    setShowClassification(false);
    document.querySelector('input[type="file"]').value = "";
  };

  const generatePDF = async () => {
    setIsGeneratingPDF(true);
    try {
      // Create a temporary div for PDF generation
      const pdfContainer = document.createElement("div");
      pdfContainer.style.width = "794px"; // A4 width in pixels (210mm)
      pdfContainer.style.padding = "20px";
      pdfContainer.style.fontFamily = "Arial, sans-serif";

      // Header
      const header = document.createElement("div");
      header.style.borderBottom = "1px solid #ccc";
      header.style.paddingBottom = "10px";
      header.style.marginBottom = "15px";
      header.style.textAlign = "center";

      // Add logo
      const logo = document.createElement("img");
      logo.src = "siit_logo.png"; // Path to the logo image
      logo.alt = "Logo";
      logo.style.width = "70px"; // Adjust the width as needed
      logo.style.marginBottom = "10px";
      header.appendChild(logo);

      // Add title and details
      const title = document.createElement("h5");
      title.style.color = "#2c3e50";
      title.style.fontSize = "1.2em";
      title.style.fontWeight = "bold";
      title.style.paddingBottom = "5px";
      title.style.margin = "0";
      title.textContent =
        "Analytical Report on Deep Learning Denoising for FTIR Microplastic Identification";
      header.appendChild(title);

      const fileName = document.createElement("p");
      fileName.style.margin = "3px 0";
      fileName.style.fontSize = "0.8em";
      fileName.innerHTML = `<strong>Source File:</strong> ${
        file?.name || "N/A"
      }`;
      header.appendChild(fileName);

      const dateIssued = document.createElement("p");
      dateIssued.style.margin = "3px 0";
      dateIssued.style.fontSize = "0.8em";
      const options = {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      };
      dateIssued.innerHTML = `<strong>Analysis Date:</strong> ${new Date().toLocaleDateString(
        undefined,
        options
      )}`;
      header.appendChild(dateIssued);

      pdfContainer.appendChild(header);

      // Content (Single Column Layout)
      const content = document.createElement("div");
      content.style.display = "flex";
      content.style.flexDirection = "column";
      content.style.alignItems = "center";
      content.style.justifyContent = "center";

      const sectionStyle = {
        marginBottom: "10px",
        paddingBottom: "10px",
        paddingTop: "5px",
        borderBottom: "1px solid #ccc", // Default border
      };

      const noBorderStyle = {
        marginBottom: "10px",
        // paddingBottom: "10px",
        paddingTop: "5px",
      };

      const headingStyle = {
        color: "#2c3e50",
        marginBottom: "3px",
        textAlign: "left",
        fontSize: "1em",
        fontWeight: "bold",
      };

      const canvasWidth = 700;
      const canvasHeight = 150;

      // Input Spectrum
      if (inputSpectrum && inputChartData) {
        const inputSection = document.createElement("div");
        Object.assign(inputSection.style, sectionStyle);
        const heading = document.createElement("h4");
        Object.assign(heading.style, headingStyle);
        heading.textContent = "Input Spectrum";
        inputSection.appendChild(heading);
        const inputCanvas = document.createElement("canvas");
        inputCanvas.width = canvasWidth;
        inputCanvas.height = canvasHeight;
        Object.assign(inputCanvas.style, {
          width: `${canvasWidth}px`,
          height: `${canvasHeight}px`,
          paddingTop: "5px", // Add 5px top padding
          paddingBottom: "5px", // Add 5px bottom padding
        });
        new ChartJS(inputCanvas, {
          type: "line",
          data: inputChartData,
          options: {
            ...chartOptions,
            animation: { duration: 0 },
            responsive: false,
            maintainAspectRatio: false,
          },
        });
        inputSection.appendChild(inputCanvas);
        content.appendChild(inputSection);
      }

      // Preprocessed Spectrum
      if (preprocessedSpectrum && preprocessedChartData) {
        const preprocessedSection = document.createElement("div");
        Object.assign(preprocessedSection.style, sectionStyle);
        const heading = document.createElement("h4");
        Object.assign(heading.style, headingStyle);
        heading.textContent = "Preprocessed Spectrum";
        preprocessedSection.appendChild(heading);
        const preprocessedCanvas = document.createElement("canvas");
        preprocessedCanvas.width = canvasWidth;
        preprocessedCanvas.height = canvasHeight;
        Object.assign(preprocessedCanvas.style, {
          width: `${canvasWidth}px`,
          height: `${canvasHeight}px`,
          paddingTop: "5px",
          paddingBottom: "5px",
        });
        new ChartJS(preprocessedCanvas, {
          type: "line",
          data: preprocessedChartData,
          options: {
            ...chartOptions,
            animation: { duration: 0 },
            responsive: false,
            maintainAspectRatio: false,
          },
        });
        preprocessedSection.appendChild(preprocessedCanvas);

        // Add preprocessing option
        const preprocessingInfo = document.createElement("p");
        preprocessingInfo.style.margin = "3px 0";
        preprocessingInfo.style.fontSize = "0.8em";
        preprocessingInfo.style.textAlign = "left";
        const preprocessingLabels = {
          noPreprocessing: "No Preprocessing",
          baselineCorrection: "Baseline Correction Only",
          minMaxNormalization: "Min-Max Normalization Only",
          baselineMinMax: "Baseline Correction + Min-Max Normalization",
        };
        preprocessingInfo.innerHTML = `<strong>Preprocessing Option:</strong> ${
          preprocessingLabels[preprocessingOption] || "N/A"
        }`;
        preprocessedSection.appendChild(preprocessingInfo);

        content.appendChild(preprocessedSection);
      }

      // Denoised Spectrum
      if (denoisedSpectrum && denoisedChartData) {
        const denoisedSection = document.createElement("div");
        Object.assign(denoisedSection.style, sectionStyle);
        const heading = document.createElement("h4");
        Object.assign(heading.style, headingStyle);
        heading.textContent = "Denoised Spectrum";
        denoisedSection.appendChild(heading);
        const denoisedCanvas = document.createElement("canvas");
        denoisedCanvas.width = canvasWidth;
        denoisedCanvas.height = canvasHeight;
        Object.assign(denoisedCanvas.style, {
          width: `${canvasWidth}px`,
          height: `${canvasHeight}px`,
          paddingTop: "5px",
          paddingBottom: "5px",
        });
        new ChartJS(denoisedCanvas, {
          type: "line",
          data: denoisedChartData,
          options: {
            ...chartOptions,
            animation: { duration: 0 },
            responsive: false,
            maintainAspectRatio: false,
          },
        });
        denoisedSection.appendChild(denoisedCanvas);

        // Add denoising model information
        const denoisingInfo = document.createElement("p");
        denoisingInfo.style.margin = "3px 0";
        denoisingInfo.style.fontSize = "0.8em";
        denoisingInfo.style.textAlign = "left";
        denoisingInfo.innerHTML = `<strong>Denoising Model:</strong> ${
          denoisingOption || "N/A"
        }`;
        denoisedSection.appendChild(denoisingInfo);

        content.appendChild(denoisedSection);
      }

      // Classification Result
      // Classification Result
      if (
        classificationResult &&
        comparisonChartData &&
        comparisonChartData.labels?.length > 0 &&
        comparisonChartData.datasets?.length > 0
      ) {
        const classificationSection = document.createElement("div");
        Object.assign(classificationSection.style, noBorderStyle);
        
        const heading = document.createElement("h4");
        Object.assign(heading.style, headingStyle);
        heading.textContent = "Classification Result";
        classificationSection.appendChild(heading);
        
        const classificationCanvas = document.createElement("canvas");
        classificationCanvas.width = canvasWidth;
        classificationCanvas.height = canvasHeight;
        Object.assign(classificationCanvas.style, {
          width: `${canvasWidth}px`,
          height: `${canvasHeight}px`,
          paddingTop: "5px",
          paddingBottom: "5px",
        });
        
        // Create a container div for the chart and legend
        const chartContainer = document.createElement("div");
        chartContainer.style.position = "relative";
        chartContainer.style.width = `${canvasWidth}px`;
        chartContainer.style.height = `${canvasHeight}px`;
        
        // Create the chart first
        new ChartJS(classificationCanvas, {
          type: "line",
          data: comparisonChartData,
          options: {
            ...chartOptionsWithLegend,
            plugins: {
              legend: {
                display: false // Hide default legend
              }
            },
            animation: { duration: 0 },
            responsive: false,
            maintainAspectRatio: false,
          }
        });
        
        // Create custom legend
        const legendDiv = document.createElement("div");
        Object.assign(legendDiv.style, {
          position: "absolute",
          top: "16px",
          right: "20px",
          backgroundColor: "rgba(255, 255, 255, 0.64)",
          padding: "8px 12px",
          lineHeight: "0.5",
          borderRadius: "2px",
          border: "1px solid #ddd",
          // boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
          zIndex: "1",
          fontSize: "0.5rem"
        });
        
        // Reference Spectrum legend item
        const refLegend = document.createElement("div");
        refLegend.style.display = "flex";
        refLegend.style.alignItems = "center";
        refLegend.style.marginBottom = "6px";
        
        const refColor = document.createElement("div");
        refColor.style.width = "20px";
        refColor.style.height = "3px";
        refColor.style.backgroundColor = "rgba(75, 192, 192, 1)";
        refColor.style.marginRight = "10px";
        
        const refText = document.createElement("span");
        refText.textContent = "Reference Spectrum";
        
        refLegend.appendChild(refColor);
        refLegend.appendChild(refText);
        
        // Denoised Spectrum legend item
        const denoisedLegend = document.createElement("div");
        denoisedLegend.style.display = "flex";
        denoisedLegend.style.alignItems = "center";
        
        const denoisedColor = document.createElement("div");
        denoisedColor.style.width = "20px";
        denoisedColor.style.height = "3px";
        denoisedColor.style.backgroundColor = "rgb(117, 2, 2)";
        denoisedColor.style.marginRight = "10px";
        
        const denoisedText = document.createElement("span");
        denoisedText.textContent = "Denoised Spectrum";
        
        denoisedLegend.appendChild(denoisedColor);
        denoisedLegend.appendChild(denoisedText);
        
        // Add legend items to legend div
        legendDiv.appendChild(refLegend);
        legendDiv.appendChild(denoisedLegend);
        
        // Add elements to container
        chartContainer.appendChild(classificationCanvas);
        chartContainer.appendChild(legendDiv);
        
        // Add container to section
        classificationSection.appendChild(chartContainer);
        
        // Keep the original result details section exactly as it was
        const resultDetails = document.createElement("div");
        resultDetails.style.fontSize = "0.8em";
        resultDetails.style.textAlign = "left";
        resultDetails.innerHTML = `
          <p style="margin: 3px 0;"><strong>Predicted Plastic Type:</strong> <span style="color:rgb(138, 14, 0);">${
            classificationResult.plastic_type || "N/A"
          }</span></p>
          <p style="margin: 3px 0;"><strong>Pearson Correlation:</strong> ${
            classificationResult.pearson_correlation?.toFixed(4) || "N/A"
          }</p>
        `;
        classificationSection.appendChild(resultDetails);
        content.appendChild(classificationSection);
      } else {
        console.warn(
          "Classification Result graph not generated due to missing or empty data."
        );
      }

      pdfContainer.appendChild(content);

      // Footer
      const footer = document.createElement("div");
      footer.style.borderTop = "1px solid #ccc";
      footer.style.paddingTop = "5px";
      footer.style.marginTop = "10px";
      footer.style.fontSize = "0.8em";
      footer.style.color = "#7f8c8d";
      footer.style.textAlign = "center";
      footer.innerHTML = `Generated by SL1 Digital Engineering Senior Project 2025`;
      pdfContainer.appendChild(footer);

      // Append the temporary container to the body
      document.body.appendChild(pdfContainer);

      // Generate PDF
      const pdf = new jsPDF("p", "pt", "a4");
      const canvas = await html2canvas(pdfContainer, {
        scale: 3,
        useCORS: true,
        allowTaint: true,
        scrollY: 0,
      });

      // Remove the temporary container
      document.body.removeChild(pdfContainer);

      const imgData = canvas.toDataURL("image/png");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save("SL1_FTIRspectraAnalysis.pdf");
    } catch (error) {
      console.error("Error generating PDF:", error);
      setError("Failed to generate PDF report");
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  useEffect(() => {
    if (!file) return;

    const fetchInputSpectrum = async () => {
      setLoadingInput(true);
      setCurrentOperation("Loading spectrum");
      setProgress(0);

      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await axios.post(
          "http://localhost:8000/api/input",
          formData,
          {
            headers: { "Content-Type": "multipart/form-data" },
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 90) / progressEvent.total
              );
              setProgress(percentCompleted);
            },
          }
        );
        setInputSpectrum(response.data.inputSpectrum);
        setProgress(100);
      } catch (error) {
        setError(error.response?.data?.detail || "Error loading spectrum");
      } finally {
        setLoadingInput(false);
        setCurrentOperation("");
      }
    };

    setError(null);
    fetchInputSpectrum();
  }, [file]);

  const fetchPreprocessedSpectrum = async () => {
    if (!file) return;

    setLoadingPreprocess(true);
    setCurrentOperation("Preprocessing");
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("preprocessing_option", preprocessingOption);

      const response = await axios.post(
        "http://localhost:8000/api/preprocess",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 90) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );
      setPreprocessedSpectrum(response.data.preprocessedSpectrum);
      setProgress(100);
    } catch (error) {
      setError(error.response?.data?.detail || "Preprocessing failed");
    } finally {
      setLoadingPreprocess(false);
      setCurrentOperation("");
    }
  };

  const fetchDenoisedSpectrum = async () => {
    if (!preprocessedSpectrum) return;

    setLoadingDenoise(true);
    setCurrentOperation("Denoising");
    setProgress(0);
    setShowClassification(false);

    try {
      const formData = new FormData();
      formData.append(
        "wavenumbers",
        JSON.stringify(preprocessedSpectrum.wavenumbers)
      );
      formData.append(
        "intensities",
        JSON.stringify(preprocessedSpectrum.intensities)
      );
      formData.append(
        "denoising_option",
        denoisingOption === "Disable" ? "Disable" : "Enable"
      );
      formData.append("model_name", denoisingOption);

      const response = await axios.post(
        "http://localhost:8000/api/denoise",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 90) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );
      const newDenoised = {
        wavenumbers: preprocessedSpectrum.wavenumbers,
        intensities: response.data.denoisedSpectrum,
      };
      setDenoisedSpectrum(newDenoised);
      setProgress(100);

      // Wait a moment to show denoised result before classification
      setTimeout(() => {
        fetchClassificationResult(newDenoised);
      }, 500);
    } catch (error) {
      setError("Denoising failed");
    } finally {
      setLoadingDenoise(false);
      setCurrentOperation("");
    }
  };

  const fetchClassificationResult = async (denoisedData = denoisedSpectrum) => {
    if (!denoisedData) return;

    setLoadingClassify(true);
    setCurrentOperation("Classifying");
    setProgress(90);

    try {
      const formData = new FormData();
      formData.append("wavenumbers", JSON.stringify(denoisedData.wavenumbers));
      formData.append("intensities", JSON.stringify(denoisedData.intensities));
      formData.append("model_name", denoisingOption);

      const response = await axios.post(
        "http://localhost:8000/api/classify",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              90 + (progressEvent.loaded * 10) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );
      setClassificationResult({
        ...response.data,
        wavenumbers: denoisedData.wavenumbers,
      });
      setProgress(100);
      setShowClassification(true);
    } catch (error) {
      setError("Classification failed");
    } finally {
      setLoadingClassify(false);
      setCurrentOperation("");
    }
  };

  return (
    <div className="container-fluid">
      <h4 className="text-center my-1">
        Deep Learning Denoising for Enhanced Microplastic FTIR Identification
      </h4>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      <style>
        {`


    .section-box {
      position: relative;
      border: 1px solid #dee2e6;
      border-radius: 0.25rem;
      padding-left: 0.8rem;
      padding-top: 0.8rem;
      padding-right: 0.8rem;
      padding-bottom: 0.2rem;
      margin-bottom: 0.8rem;
      font-size: 0.7rem;
      
    }

    .section-title {
      position: absolute;
      top: -0.75rem;
      left: 50%;
      transform: translateX(-50%);
      background: white;
      font-weight: 600;
      font-size: 0.95rem;
      color: #2c3e50;
      z-index: 1;
      white-space: nowrap;
    }


    .section-actions {
      display: flex;
      justify-content: flex-end;
      margin-right: 0.8rem;
      margin-bottom: 0.1rem;
    }

    .section-actions button {
      align-items: end;
      margin-top: 0.5rem;
      margin-bottom: 0.8rem;
      margin-right: -0.5rem;
      font-size: 0.5rem;
      font-weight: 600;
      padding: 0.3rem 1rem;
      color: black;
      border: none;
      border-radius: 0.25rem;
      cursor: pointer;
      box-shadow: 2px 2px 6px 2px rgba(0, 0, 0, 0.1);
      transition: background-color 0.3s ease;
    }
    .section-actions button:hover {
      background-color:rgb(61, 61, 61);
      color: white;
    }
    .section-actions button:disabled {
      background-color: #e9ecef;
      color: #6c757d;
      cursor: pointer;
      box-shadow: none;
    }
    .section-actions button:disabled:hover {
      background-color: #e9ecef;
      color: #6c757d;
      cursor: pointer;
      box-shadow: none;
    }
    
    .form-check {
      display: flex;
      font-weight: 600;
      font-size: 0.7rem;
      // line-height: 0; 
      margin-bottom: -0.3rem; 
    }

    .form-check-input {
      margin-right: 0.5rem;
    }   
    
    .form-check-input:checked {
      background-color: gray;
      border-color: gray;
    }

    .form-check-input:focus {
      border-color: #ced4da; 
      box-shadow: 0 0 0 0.25rem rgba(108, 117, 125, 0.25); 
    }

    .btn-clear {
      align-items: center;
      background-color: rgb(201, 73, 73);
      padding: 5px;
      border: none;
      border-radius: 5px;
      color: rgb(255, 255, 255);
      font-size: x-small;
      cursor: pointer;
      box-shadow: 2px 2px 6px 0 rgba(0, 0, 0, 0.1);
      transition: background-color 0.3s ease;
    }
    .btn-clear:hover {
      background-color:rgb(0, 0, 0);
      color: white;
      cursor: pointer;
    }

    .btn-save {
      align-items: center;
      background-color: #28a745;
      padding: 5px 10px;
      border: none;
      border-radius: 5px;
      color: rgb(255, 255, 255);
      font-size: small;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 2px 2px 6px 0 rgba(0, 0, 0, 0.1);
      transition: background-color 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .btn-save:hover {
      background-color: rgb(0, 0, 0);
      color: white;
    }

    .chart-display-area {
        height: 240px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .classification-result-content {
        display: flex;
        flex-direction: column;
        height: 100%;
        padding-top: 0.9rem;
    }

    .classification-chart-area {
        height: 200px;
        overflow: hidden;
        flex-shrink: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    .classification-text-results {
        text-align: start;
        // line-height: 1;
        justify-content: center;
        align-items: center;
        align-self: center;
        flex-shrink: 0;
        margin: 0.5rem;
        height: 40px;
        font-size: 0.7rem;
        font-weight: 600;
    }
 `}
      </style>

      {/* Row for File Input and Save Report Button */}
      <div className="row mb-3 align-items-center">
        <div className="col-md-11">
          <input
            type="file"
            onChange={handleFileChange}
            className="form-control"
            accept=".csv"
          />
        </div>
        <div className="col-md-1">
          <button
            onClick={generatePDF}
            className="btn-save"
            disabled={!classificationResult || isGeneratingPDF}
          >
            {isGeneratingPDF ? (
              <>
                <Spinner
                  as="span"
                  animation="border"
                  size="sm"
                  className="me-2"
                />
                Saving...
              </>
            ) : (
              <>
                <img
                  src="download (1).png"
                  alt="Save"
                  style={{
                    width: "20px",
                    height: "20px",
                    marginRight: "5px",
                  }}
                />
                Save
              </>
            )}
          </button>
        </div>
      </div>

      <div className="row">
        <div className="col-md-3">
          <div className="p-10">
            {/* Preprocessing Box */}
            <div className="section-box">
              <div className="section-title">Preprocessing</div>
              {[
                { value: "noPreprocessing", label: "No Preprocessing" },
                {
                  value: "baselineCorrection",
                  label: "Baseline Correction Only",
                },
                {
                  value: "minMaxNormalization",
                  label: "Min-Max Normalization Only",
                },
                {
                  value: "baselineMinMax",
                  label: "Baseline Correction + Min-Max Normalization",
                },
              ].map(({ value, label }) => (
                <div className="form-check" key={value}>
                  <input
                    className="form-check-input"
                    type="radio"
                    accent-color="gray"
                    name="preprocessingOption"
                    id={value}
                    value={value}
                    checked={preprocessingOption === value}
                    onChange={(e) => setPreprocessingOption(e.target.value)}
                  />
                  <label className="form-check-label" htmlFor={value}>
                    {label}
                  </label>
                </div>
              ))}
              <div className="section-actions">
                <button
                  onClick={fetchPreprocessedSpectrum}
                  className="btn"
                  // disabled={loadingPreprocess || !file}
                >
                  Apply
                </button>
              </div>
            </div>

            {/* Denoising Box */}
            <div className="section-box">
              <div className="section-title">Denoising Model</div>
              {[
                "Disable",
                "Autoencoder",
                "Unet",
                "CNN-Autoencoder (CNNAE)",
                "CNNAE-MobileNet",
                "CNNAE-ResNet50",
                "CNNAE-InceptionV3",
                "CNNAE-InceptionResNet",
                "CNNAE-Xception",
              ].map((option, index) => (
                <div className="form-check" key={option}>
                  <input
                    className="form-check-input"
                    type="radio"
                    name="denoisingOption"
                    id={`denoise-${index}`}
                    value={option}
                    checked={denoisingOption === option}
                    onChange={(e) => setDenoisingOption(e.target.value)}
                  />
                  <label
                    className="form-check-label"
                    htmlFor={`denoise-${index}`}
                  >
                    {option}
                  </label>
                </div>
              ))}
              <div className="section-actions">
                <button
                  onClick={fetchDenoisedSpectrum}
                  className="btn"
                  // disabled={loadingDenoise || !preprocessedSpectrum}
                >
                  Run
                </button>
              </div>
            </div>

            {/* Classification Box */}
            <div className="section-box">
              <div className="section-title">Classification Result Details</div>
              {loadingClassify ? (
                <div className="classification-text-results">
                  <p>Classifying the spectrum...</p>
                </div>
              ) : showClassification && classificationResult ? (
                <div className="classification-text-results">
                  <p className="mb-1">
                    Predicted Plastic Type:{" "}
                    <span className="text-danger fw-bold">
                      {classificationResult.plastic_type || "N/A"}
                    </span>
                  </p>
                  <p className="mb-0">
                    Pearson Correlation:{" "}
                    {classificationResult.pearson_correlation?.toFixed(4) ||
                      "N/A"}
                  </p>
                </div>
              ) : (
                <div className="classification-text-results text-center">
                  <p>No classification result available</p>
                </div>
              )}
            </div>

            {/* Clear Button */}
            <button
              onClick={handleClear}
              className="btn-clear btn-sm btn-outline-danger w-100 mb-2"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="col-md-9">
          <div className="row">
            {/* Input Spectrum Box */}
            <div className="col-md-6">
              <div className="section-box">
                <div className="section-title">Input Spectrum</div>
                <div className="chart-display-area">
                  {loadingInput ? (
                    <div className="d-flex flex-column justify-content-center align-items-center h-100">
                      <Spinner animation="border" />
                      <p className="mt-2">Loading spectrum...</p>
                      <ProgressBar
                        now={progress}
                        animated
                        striped
                        className="w-100 mt-2"
                        style={{ height: "6px", transition: "width 0.3s ease" }}
                      />
                    </div>
                  ) : inputSpectrum ? (
                    <Line data={inputChartData} options={chartOptions} />
                  ) : (
                    <div className="centered-text">
                      <p>No data to display</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Preprocessed Spectrum Box */}
            <div className="col-md-6">
              <div className="section-box">
                <div className="section-title">Preprocessed Spectrum</div>
                <div className="chart-display-area">
                  {loadingPreprocess ? (
                    <div className="d-flex flex-column justify-content-center align-items-center h-100">
                      <Spinner animation="border" />
                      <p className="mt-2">Preprocessing...</p>
                      <ProgressBar
                        now={progress}
                        animated
                        striped
                        className="w-100 mt-2"
                        style={{ height: "6px", transition: "width 0.3s ease" }}
                      />
                    </div>
                  ) : preprocessedSpectrum ? (
                    <Line data={preprocessedChartData} options={chartOptions} />
                  ) : (
                    <div className="centered-text">
                      <p>No data to display</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Denoised Spectrum Box */}
            <div className="col-md-6">
              <div className="section-box">
                <div className="section-title">Denoised Spectrum</div>
                <div className="chart-display-area">
                  {loadingDenoise ? (
                    <div className="d-flex flex-column justify-content-center align-items-center h-100">
                      <Spinner animation="border" />
                      <p className="mt-2">Denoising...</p>
                      <ProgressBar
                        now={progress}
                        animated
                        striped
                        className="w-100 mt-2"
                        style={{ height: "6px", transition: "width 0.3s ease" }}
                      />
                    </div>
                  ) : denoisedSpectrum ? (
                    <Line data={denoisedChartData} options={chartOptions} />
                  ) : (
                    <div className="centered-text">
                      <p>No data to display</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Classification Result Box */}
            <div className="col-md-6">
              <div className="section-box">
                <div className="section-title">Classification Result</div>
                <div
                  className="chart-display-area"
                  style={{ position: "relative" }}
                >
                  {loadingClassify ? (
                    <div className="d-flex flex-column justify-content-center align-items-center h-100">
                      <Spinner animation="border" />
                      <p className="mt-2">Classifying...</p>
                      <ProgressBar
                        now={progress}
                        animated
                        striped
                        className="w-100 mt-2"
                        style={{ height: "6px", transition: "width 0.3s ease" }}
                      />
                    </div>
                  ) : showClassification && classificationResult ? (
                    <>
                      {/* Chart with Custom Legend */}
                      <div style={{ position: "relative", width: "100%", height: "100%" }}>
                        <Line
                          data={comparisonChartData}
                          options={chartOptions}
                        />
                        {/* Custom HTML Legend */}
                        <div
                          style={{
                            position: "absolute",
                            top: "12px",
                            right: ["Acrylic", "Cellulose", "PBAT", "PBS", "PC", "PEEK", "PET", "PLA", "PMMA", "POM", "PS", "PTFE", "ENR", "PEI"].includes(classificationResult.plastic_type) ? "20px" : "auto",
                            left: ["EPDM", "HDPE", "LDPE", "Nylon", "PVA", "PVC", "PP"].includes(classificationResult.plastic_type) ? "56px" : "auto",
                            backgroundColor: "rgba(255, 255, 255, 0.9)",
                            padding: "8px 12px",
                            lineHeight: "0.5",
                            borderRadius: "2px",
                            border: "1px solid #ddd",
                            boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
                            zIndex: 1,
                          }}
                        >
                          <div
                            style={{ display: "flex", alignItems: "center" }}
                          >
                            <div
                              style={{
                                width: "20px",
                                height: "3px",
                                backgroundColor: "rgba(75, 192, 192, 1)",
                                marginRight: "10px",
                              }}
                            ></div>
                            <span style={{ fontSize: "0.7rem" }}>
                              Reference
                            </span>
                          </div>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              marginTop: "6px",
                            }}
                          >
                            <div
                              style={{
                                width: "20px",
                                height: "3px",
                                backgroundColor: "rgb(117, 2, 2)",
                                marginRight: "10px",
                              }}
                            ></div>
                            <span style={{ fontSize: "0.7rem" }}>
                              Denoised
                            </span>
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="centered-text">
                      <p>No data to display</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
