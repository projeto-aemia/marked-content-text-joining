Attribute VB_Name = "NewMacros"
Option Explicit

Sub SaveAllAsPDF()

    Dim strSourcePath As String
    Dim strDestinationPath As String
    Dim strFilename As String
    Dim strDocName As String
    Dim strPDFPath As String

    Dim oDoc As Document
    Dim fDialog As FileDialog
    Dim fso As Object

    Dim converted As Long
    Dim skipped As Long
    Dim failed As Long

    Dim errorDetails As String
    Dim logFilePath As String
    Dim logFileNum As Integer

    ' Array to hold file list so Dir$ never breaks
    Dim fileList() As String
    Dim fileCount As Long
    Dim i As Long

    ' Store original options to restore later
    Dim origUpdateFields As Boolean
    Dim origUpdateLinks As Boolean
    Dim origBackgroundPrint As Boolean

    Set fso = CreateObject("Scripting.FileSystemObject")

    ' Select source folder
    Set fDialog = Application.FileDialog(msoFileDialogFolderPicker)
    With fDialog
        .Title = "Select folder containing DOCX files"
        .AllowMultiSelect = False
        If .Show <> -1 Then Exit Sub
    End With

    strSourcePath = fDialog.SelectedItems(1)
    If Right(strSourcePath, 1) <> "\" Then strSourcePath = strSourcePath & "\"

    ' Select destination folder
    With fDialog
        .Title = "Select folder to save PDF files"
        If .Show <> -1 Then Exit Sub
    End With

    strDestinationPath = fDialog.SelectedItems(1)
    If Right(strDestinationPath, 1) <> "\" Then strDestinationPath = strDestinationPath & "\"

    ' 1. Cache all matching files into memory first
    fileCount = 0
    strFilename = Dir$(strSourcePath & "*.docx")

    Do While Len(strFilename) > 0
        If Left(strFilename, 2) <> "~$" Then
            fileCount = fileCount + 1
            ReDim Preserve fileList(1 To fileCount)
            fileList(fileCount) = strFilename
        End If
        strFilename = Dir$()
    Loop

    If fileCount = 0 Then
        MsgBox "No DOCX files were found in the source folder.", vbInformation, "Batch PDF Conversion"
        Exit Sub
    End If

    ' Save current options
    origUpdateFields = Application.Options.UpdateFieldsAtPrint
    origUpdateLinks = Application.Options.UpdateLinksAtOpen
    origBackgroundPrint = Application.Options.PrintBackground

    ' Enforce strict silent mode
    Application.Options.UpdateFieldsAtPrint = False
    Application.Options.UpdateLinksAtOpen = False
    Application.Options.PrintBackground = False
    Application.DisplayAlerts = wdAlertsNone
    Application.ScreenUpdating = False

    ' 2. Loop through the cached file list
    For i = 1 To fileCount
        strFilename = fileList(i)
        strDocName = Left(strFilename, InStrRev(strFilename, ".") - 1)
        strPDFPath = strDestinationPath & strDocName & ".pdf"

        ' Skip already converted
        If fso.FileExists(strPDFPath) Then
            skipped = skipped + 1
        Else
            Set oDoc = Nothing

            ' Open step with localized error capture
            On Error Resume Next
            Set oDoc = Documents.Open( _
                FileName:=strSourcePath & strFilename, _
                ConfirmConversions:=False, _
                ReadOnly:=True, _
                AddToRecentFiles:=False, _
                Visible:=False, _
                NoEncodingDialog:=True)

            If Err.Number <> 0 Or oDoc Is Nothing Then
                failed = failed + 1
                errorDetails = errorDetails & vbCrLf & strFilename & " [Open Error " & Err.Number & ": " & Err.Description & "]"
                Err.Clear
                GoTo LoopNext
            End If

            ' Export step with localized error capture
            oDoc.ExportAsFixedFormat _
                OutputFileName:=strPDFPath, _
                ExportFormat:=wdExportFormatPDF, _
                OpenAfterExport:=False, _
                OptimizeFor:=wdExportOptimizeForPrint, _
                CreateBookmarks:=wdExportCreateNoBookmarks

            If Err.Number <> 0 Then
                failed = failed + 1
                errorDetails = errorDetails & vbCrLf & strFilename & " [Export Error " & Err.Number & ": " & Err.Description & "]"
                Err.Clear

                ' Delete corrupt/zero-byte PDF if Word generated a stub
                If fso.FileExists(strPDFPath) Then
                    fso.DeleteFile strPDFPath, True
                End If
            Else
                converted = converted + 1
            End If

            ' Close document safely
            oDoc.Close SaveChanges:=wdDoNotSaveChanges
            Set oDoc = Nothing
            Err.Clear
            On Error GoTo 0
        End If

LoopNext:
        ' Ensure document cleanup if Open succeeded but Export failed
        On Error Resume Next
        If Not oDoc Is Nothing Then
            oDoc.Close SaveChanges:=wdDoNotSaveChanges
            Set oDoc = Nothing
        End If
        Err.Clear
        On Error GoTo 0
    Next i

CleanExit:
    ' Restore Word settings
    Application.Options.UpdateFieldsAtPrint = origUpdateFields
    Application.Options.UpdateLinksAtOpen = origUpdateLinks
    Application.Options.PrintBackground = origBackgroundPrint
    Application.DisplayAlerts = wdAlertsAll
    Application.ScreenUpdating = True

    ' Output error log if any file failed
    If failed > 0 Then
        logFilePath = strDestinationPath & "_conversion_errors.txt"
        On Error Resume Next
        logFileNum = FreeFile
        Open logFilePath For Output As #logFileNum
        Print #logFileNum, "PDF Conversion Error Log - " & Now
        Print #logFileNum, "========================================"
        Print #logFileNum, errorDetails
        Close #logFileNum
        On Error GoTo 0
    End If

    ' Display final report
    MsgBox "Batch PDF conversion completed!" & vbCrLf & vbCrLf & _
           "Successfully converted: " & converted & vbCrLf & _
           "Skipped (PDF already exists): " & skipped & vbCrLf & _
           "Failed: " & failed & _
           IIf(failed > 0, vbCrLf & "(Logged in _conversion_errors.txt)", "") & vbCrLf & vbCrLf & _
           "PDF destination:" & vbCrLf & strDestinationPath, _
           vbInformation, "PDF Conversion Summary"

    Set fso = Nothing

End Sub


