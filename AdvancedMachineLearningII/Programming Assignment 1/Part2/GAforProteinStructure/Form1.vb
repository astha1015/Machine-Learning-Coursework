Imports System.Math
Imports System.Drawing
Imports System.Drawing.Graphics

Public Class Form1

    Dim ValidFolding As Integer
    Dim popSize As Integer = 200        ' set population size to 200
    Dim proteinLength As Integer        ' set protein sequence length
    Dim HPModel() As Integer            ' stores the position of each Hydrophobic "H" residues
    Dim max_generations As Integer = 20000   ' maximum generations if obtained fitness is not greater than or equal to target fitness
    Dim target_fitness As Integer           ' set target fitness by parsing the input file
    Dim ElitRate = 0.05F                    ' set elit rate to 5%
    Dim CrossOverRate = 0.9F                ' set crossover rate to 90%
    Dim MutationRate = 0.05F                ' set mutation rate of 5%

    Dim CurrentPosNewPopulation As Integer = 0  ' initialize current position of new population to zero

    ' Structure holds the "X" and "Y" coordinates and Fitness of the given sequence of residues
    Public Structure genotype
        Dim Fitness As Integer
        Dim X() As Integer
        ' Dim Y(1 To 64) As Integer
        Dim Y() As Integer
    End Structure

    'Array of population and new population
    Dim population(popSize - 1) As genotype          'stores current population
    Dim newpopulation(popSize - 1) As genotype       'stores new population

    Dim genoTemp1 As genotype



    ' After the button is clicked, window to select file will appear and when input file is selected
    ' Protein sequence line and fitness line are parsed and set to the variable
    ' At the same time an array "HPModel" is created which store the index of the hydrophobic residues in the given sequence

    Private Sub Button1_Click(sender As Object, e As EventArgs) Handles Button1.Click

        If OpenFileDialog1.ShowDialog = DialogResult.OK Then
            TextBox1.Text = OpenFileDialog1.FileName

            Dim fileReader As System.IO.StreamReader
            fileReader =
            My.Computer.FileSystem.OpenTextFileReader(OpenFileDialog1.FileName)
            Dim stringReader As String

            stringReader = fileReader.ReadLine()
            proteinLength = stringReader.Length
            target_fitness = -1 * (fileReader.ReadLine())
            'MsgBox("The first line of the file is " & stringReader)
            'MsgBox("String Length" & proteinLength)

            Dim i As Integer
            Dim hCount As Integer = 0
            Dim pCount As Integer = 0
            Dim resArray(proteinLength - 1) As Char
            resArray = stringReader.ToCharArray

            For i = 0 To proteinLength - 1

                If (resArray(i) = "h") Then

                    hCount += 1

                ElseIf (resArray(i) = "p") Then
                    pCount += 1

                End If


            Next

            ReDim HPModel(hCount - 1)
            Dim counter As Integer = 0
            For i = 0 To proteinLength - 1

                If (resArray(i) = "h") Then

                    HPModel(counter) = i
                    counter += 1

                End If


            Next

        End If
    End Sub

    ' As the "Start GA" button is clicked, this function parses the value from text boxes at user end and initializes the corresponding parameters of
    ' genetice algorithm. It then calls the method GeneticAlgorithm() to start the actual GA process.
    Private Sub Button2_Click(sender As Object, e As EventArgs) Handles Button2.Click

        ElitRate = TextBox4.Text / 100
        CrossOverRate = TextBox5.Text / 100
        MutationRate = TextBox6.Text / 100
        popSize = TextBox3.Text
        target_fitness = TextBox7.Text
        GeneticAlgorithm()
    End Sub

    ' This is where the actual GA computation occurs. It sets the required parameters and calls the sub methods ( InitializePopulation, sortPopWithFitness, doElit, crossover, fillTheRest, Mutation)
    ' and iterates unless the obtained fitness is equal to the total fitness or unless the number of generation reaches the max generation
    Private Function GeneticAlgorithm()
        Me.TextBox2.Refresh()
        GeneticAlgorithm = Nothing
        Dim generations As Integer = 0

        Dim obtained_fitness = 0.0F

        ReDim genoTemp1.X(proteinLength)
        ReDim genoTemp1.Y(proteinLength)
        InitializePopulation()
        sortPopWithFitness(population, genoTemp1)

        While (generations < max_generations) And (obtained_fitness < target_fitness)
            generations += 1
            Dim elit_index As Integer
            CurrentPosNewPopulation = doElit(population, newpopulation)
            CurrentPosNewPopulation += 1
            elit_index = CurrentPosNewPopulation

            ' Crossover Starts
            Dim rdm As New Random()
            Dim totalFitness = 0.0F
            Dim rand1 = 0.0F
            Dim rand2 = 0.0F
            Dim firstSelectionIndex As Integer
            Dim secondSelectionIndex As Integer
            Dim cross_over_point As Integer
            Dim num_of_Crossover = CInt((CrossOverRate * (popSize - 1)) / 2)

            Dim i As Integer
            For i = 0 To popSize - 1
                totalFitness += population(i).Fitness
            Next

            Dim j As Integer

            For j = 0 To num_of_Crossover - 1

                rand1 = rdm.Next(1, (totalFitness + 1))
                firstSelectionIndex = 0
                While (rand1 > 0)

                    rand1 = rand1 - population(firstSelectionIndex).Fitness
                    firstSelectionIndex += 1

                End While

                rand2 = rdm.Next(1, (totalFitness + 1))
                secondSelectionIndex = 0
                While (rand2 > 0)

                    rand2 = rand2 - population(secondSelectionIndex).Fitness
                    secondSelectionIndex += 1

                End While

                cross_over_point = rdm.Next(1, (proteinLength - 2))

                If (CrossOver(firstSelectionIndex - 1, secondSelectionIndex - 1, cross_over_point) = 1 And CrossOver(secondSelectionIndex - 1, firstSelectionIndex - 1, cross_over_point) = 1) Then

                    CurrentPosNewPopulation = CurrentPosNewPopulation + CrossOver(firstSelectionIndex - 1, secondSelectionIndex - 1, cross_over_point) ' once call with first selection index
                    CurrentPosNewPopulation = CurrentPosNewPopulation + CrossOver(secondSelectionIndex - 1, firstSelectionIndex - 1, cross_over_point)  ' then call with second selection index
                Else
                    j = j - 1
                    Continue For
                End If


            Next

            '  End of Crossover 



            ' filling in rest
            Dim fill_rest_count As Integer = 0
            Dim p As Integer

            For p = CurrentPosNewPopulation + 1 To popSize - 1

                NewRandomOrientation(p)
                fill_rest_count += 1

            Next

            ' End of filling in rest



            ' Mutation starts here
            Try
                Dim num_of_mutation = CInt(MutationRate * (popSize - 1))
                Dim num_of_Elit = CInt(ElitRate * (popSize - 1))
                Dim chromosome_index_to_mutate As Integer = 0
                Dim mutation_index As Integer = 0
                Dim q As Integer = 0
                Dim success As Integer

                For q = 0 To num_of_mutation - 1

                    chromosome_index_to_mutate = rdm.Next(num_of_Elit, popSize)


                    mutation_index = rdm.Next(1, (proteinLength - 2))

                    success = Mutation(chromosome_index_to_mutate, mutation_index)
                    If success = 0 And q > 0 Then
                        q = q - 1

                    ElseIf success = 1 Then
                        CurrentPosNewPopulation = CurrentPosNewPopulation + 1
                        If CurrentPosNewPopulation = popSize - 1 Then
                            Exit For
                        End If
                    End If

                Next
            Catch ex As Exception
                Console.WriteLine(ex.Message)
                MsgBox("The sum percentage of Elite, Crossover and Mutation should be 100% but the current position of new population before mutation is = " & CurrentPosNewPopulation)
                Process.GetCurrentProcess.Kill()
            End Try
            ' Mutation Ends


            ' Population Swapping
            Dim tempPopulation() As genotype = population
            population = newpopulation
            newpopulation = tempPopulation

            ' Ends 


            'Calculating Fintness after swaping

            Dim r As Integer
            For r = elit_index To popSize - 1

                population(r).Fitness = ComputeFitness(r)
                'DrawStructurePictureBox1(r, population)

            Next

            ' Ends 


            sortPopWithFitness(population, genoTemp1)           ' sort the chromosomes along with their fitness in descending order 
            obtained_fitness = population(0).Fitness            ' fitness of the first chromosome in a population will be the best fitness
            Console.Write("Fitness at each generation = ")
            Console.WriteLine(-1 * obtained_fitness)            ' finally display the fitness by multiplying it by "-1" on console 

            ' Printing the fitness value in the Textbox for best chromosome in each iteration ******************************

            Dim myGraphics As Graphics
            myGraphics = Graphics.FromHwnd(Me.TextBox2.Handle)
            myGraphics.Clear(Color.White)
            myGraphics.DrawString(-obtained_fitness.ToString, New Font("Tahoma", 11), Brushes.Black, New Point(20, 0))

            ' ******************************* End Printing the value of fitness in the Textbox for best chromosome in each iteration ******************************

            ' ****************************** Draw the structure in PictureBox2 ********************************
            DrawStructurePictureBox2(0, population)

            ' ****************************** End Drawing the structure in PictureBox2 ********************************

        End While

        obtained_fitness = population(0).Fitness                ' At the end of total iteration the first chromosome of the population is the best structure
        Console.Write("Final Obtained Fitness = ")
        Console.WriteLine(-1 * obtained_fitness)
        ' *******************************  Print the value of fitness in the Textbox for best chromosome in each iteration ******************************

        Dim finalGraphics As Graphics
        finalGraphics = Graphics.FromHwnd(Me.TextBox2.Handle)
        'finalGraphics.Clear(Color.White)
        finalGraphics.DrawString(-obtained_fitness.ToString, New Font("Tahoma", 11), Brushes.Black, New Point(20, 0))

        ' ******************************* End Printing the value of fitness in the Textbox for best chromosome in each iteration ******************************
        DrawStructurePictureBox2(0, population)

    End Function

    ' ***********************************************************  End of Genetice Algorithm ***************************************************************************************************



    ' ========================================== Start Mutation ==============================
    Function Mutation(i As Long, n As Integer) As Long
        Dim id As Long
        Dim a As Long
        Dim b As Long
        Dim A_Limit As Long
        Dim choice As Long
        Dim Collision As Long
        Dim rdm As New Random

        Dim k As Long
        Dim z As Long
        Dim p As Long


        Dim Ary(2) As Long

        id = CurrentPosNewPopulation
        ReDim newpopulation(id).X(proteinLength - 1)
        ReDim newpopulation(id).Y(proteinLength - 1)

        ' possible rotations 90ß,180ß,270ß
        '           index       0   1    2
        '


        Ary(0) = 0
        Ary(1) = 1
        Ary(2) = 2
        A_Limit = 2

        a = population(i).X(n)          '/* (a, b) rotating point */
        b = population(i).Y(n)

        Do
            Collision = 0
            If (A_Limit > 0) Then

                choice = rdm.Next(0, 2)
            Else
                choice = A_Limit
            End If


            p = Ary(choice)
            For k = choice To A_Limit - 1
                Ary(k) = Ary(k + 1)
            Next k

            A_Limit = A_Limit - 1

            For k = n + 1 To proteinLength - 1
                Select Case p

                    Case 1
                        newpopulation(id).X(k) = a + b - population(i).Y(k)       '/* X' = (a+b)-Y  */
                        newpopulation(id).Y(k) = population(i).X(k) + b - a       '/* Y' = (X+b)-a  */
                    Case 2
                        newpopulation(id).X(k) = 2 * a - population(i).X(k)       '/* X' = (2a - X) */
                        newpopulation(id).Y(k) = 2 * b - population(i).Y(k)       '/* Y' = (2b-Y)   */
                    Case 3
                        newpopulation(id).X(k) = population(i).Y(k) + a - b       '/* X' =  Y+a-b   */
                        newpopulation(id).Y(k) = a + b - population(i).X(k)       '/* Y' =  (a+b)-X */
                End Select

                For z = 0 To n

                    If ((newpopulation(id).X(k) = population(i).X(z)) And (newpopulation(id).Y(k) = population(i).Y(z))) Then
                        Collision = 1
                        'MutationInternalFailCount = MutationInternalFailCount + 1
                        'MutationCollisionCount = MutationCollisionCount + 1
                        GoTo MyJump
                    End If
                Next z
            Next k

            If (Collision = 0) Then
                A_Limit = 0
            End If
MyJump:
        Loop Until A_Limit = 0

        If (Collision = 0) Then

            For k = 0 To n
                newpopulation(id).X(k) = population(i).X(k)
                newpopulation(id).Y(k) = population(i).Y(k)
            Next k


            Mutation = 1
        Else
            'MutationFailCount = MutationFailCount + 1
            Mutation = 0
        End If

    End Function

    ' =============================== End of Mutation ===================================================



    ' ================================ Start of CrossOver ============================================
    Function CrossOver(i As Long, j As Long, n As Integer) As Long

        Dim PrevDirection, k, z, p As Long
        Dim temp1, temp2, temp3, Collision, dx, dy, Step2 As Long
        Dim id As Long
        Dim rdm As New Random()


        Dim a(3) As Long
        Dim Ax(3) As Long
        Dim Ay(3) As Long

        id = CurrentPosNewPopulation
        ReDim newpopulation(id).X(proteinLength - 1)
        ReDim newpopulation(id).Y(proteinLength - 1)

        '/* Detect Previous Direction */
        If (population(i).X(n) = population(i).X(n - 1)) Then
            p = population(i).Y(n - 1) - population(i).Y(n)
            If (p = 1) Then
                PrevDirection = 2
            Else
                PrevDirection = 3
            End If

        Else
            p = population(i).X(n - 1) - population(i).X(n)
            If (p = 1) Then
                PrevDirection = 0
            Else
                PrevDirection = 1
            End If
        End If


        Select Case PrevDirection
            Case 0
                Ax(0) = -1
                Ay(0) = 0
                Ax(1) = 0
                Ay(1) = 1
                Ax(2) = 0
                Ay(2) = -1
            Case 1
                Ax(0) = 1
                Ay(0) = 0
                Ax(1) = 0
                Ay(1) = 1
                Ax(2) = 0
                Ay(2) = -1
            Case 2
                Ax(0) = 1
                Ay(0) = 0
                Ax(1) = -1
                Ay(1) = 0
                Ax(2) = 0
                Ay(2) = -1

            Case 3
                Ax(0) = 1
                Ay(0) = 0
                Ax(1) = -1
                Ay(1) = 0
                Ax(2) = 0
                Ay(2) = 1
        End Select


        temp1 = rdm.Next(0, 3)
        newpopulation(id).X(n + 1) = population(i).X(n) + Ax(temp1)
        newpopulation(id).Y(n + 1) = population(i).Y(n) + Ay(temp1)
        Collision = 0

        dx = newpopulation(id).X(n + 1) - population(j).X(n + 1)
        dy = newpopulation(id).Y(n + 1) - population(j).Y(n + 1)

        For k = n + 1 To proteinLength - 1
            newpopulation(id).X(k) = population(j).X(k) + dx

            newpopulation(id).Y(k) = population(j).Y(k) + dy

            For z = 0 To n
                If ((newpopulation(id).X(k) = population(i).X(z)) And (newpopulation(id).Y(k) = population(i).Y(z))) Then
                    Collision = 1
                    'CrossoverInternalFailCount = CrossoverInternalFailCount + 1
                    'CrossoverCollisionCount = CrossoverCollisionCount + 1
                    GoTo MyOut1
                End If
            Next z
        Next k

MyOut1:
        If (Collision = 1) Then         '/* ======> Second try ==== */
            Collision = 0
            Step2 = 4 - temp1
            Select Case Step2
                Case 2
                    If rdm.Next(0, 2) = 0 Then
                        temp2 = 0
                    Else
                        temp2 = 1
                    End If
                Case 3
                    If rdm.Next(0, 2) = 0 Then
                        temp2 = 0
                    Else
                        temp2 = 2
                    End If
                Case 4
                    If rdm.Next(0, 2) = 0 Then
                        temp2 = 1
                    Else
                        temp2 = 2
                    End If
            End Select

            temp3 = 4 - (temp1 + temp2)
            newpopulation(id).X(n + 1) = population(i).X(n) + Ax(temp2)
            newpopulation(id).Y(n + 1) = population(i).Y(n) + Ay(temp2)
            dx = newpopulation(id).X(n + 1) - population(j).X(n + 1)
            dy = newpopulation(id).Y(n + 1) - population(j).Y(n + 1)

            For k = n + 1 To proteinLength - 1

                newpopulation(id).X(k) = population(j).X(k) + dx
                newpopulation(id).Y(k) = population(j).Y(k) + dy

                For z = 0 To n
                    If ((newpopulation(id).X(k) = population(i).X(z)) And (newpopulation(id).Y(k) = population(i).Y(z))) Then
                        Collision = 1
                        'CrossoverCollisionCount = CrossoverCollisionCount + 1
                        'CrossoverInternalFailCount = CrossoverInternalFailCount + 1
                        GoTo MyOut2
                    End If
                Next z
            Next k

MyOut2:
            If (Collision = 1) Then
                Collision = 0
                newpopulation(id).X(n + 1) = population(i).X(n) + Ax(temp3)
                newpopulation(id).Y(n + 1) = population(i).Y(n) + Ay(temp3)
                dx = newpopulation(id).X(n + 1) - population(j).X(n + 1)
                dy = newpopulation(id).Y(n + 1) - population(j).Y(n + 1)
                For k = n + 1 To proteinLength - 1
                    newpopulation(id).X(k) = population(j).X(k) + dx
                    newpopulation(id).Y(k) = population(j).Y(k) + dy
                    For z = 0 To n
                        If ((newpopulation(id).X(k) = population(i).X(z)) And (newpopulation(id).Y(k) = population(i).Y(z))) Then
                            Collision = 1
                            'CrossoverInternalFailCount = CrossoverInternalFailCount + 1
                            'CrossoverCollisionCount = CrossoverCollisionCount + 1
                            GoTo MyOut3
                        End If
                    Next z
                Next k
            End If '/* 3rd try if ends */
        End If '/* 2nd try if ends */

MyOut3:
        If Collision = 0 Then
            '   CrossoverSuccessCount = CrossoverSuccessCount + 1
            For k = 0 To n
                newpopulation(id).X(k) = population(i).X(k)
                newpopulation(id).Y(k) = population(i).Y(k)
            Next k

            CrossOver = 1

        End If

        Return CrossOver

    End Function

    ' ================================ End of CrossOver ============================================


    ' ================================ Start of Elite ============================================
    Private Function doElit(ByVal population() As genotype, ByVal newpopulation() As genotype) As Integer

        Dim num_elit = CInt(ElitRate * popSize)
        Dim i As Integer

        If (num_elit = 0) Then

            Console.WriteLine("either population size is too small or elit rate is too small")

        End If
        For i = 0 To num_elit
            newpopulation(i) = population(i)

        Next

        Return (num_elit - 1)   'because index starts from 0

    End Function

    ' ================================ End of Elite ============================================


    ' ================================ Start of sorting chromosome along with their fitness ============================================
    Private Function sortPopWithFitness(ByVal population() As genotype, ByVal genoTemp1 As genotype)
        sortPopWithFitness = Nothing
        Dim c As Integer
        Dim d As Integer
        For c = 0 To popSize - 2
            For d = c + 1 To popSize - 1

                If (population(c).Fitness < population(d).Fitness) Then

                    genoTemp1 = population(d)
                    population(d) = population(c)
                    population(c) = genoTemp1

                End If

            Next
        Next

    End Function
    ' ================================ End of sorting chromosome along with their fitness ============================================

    ' ================================ Start of Initializing population ============================================
    Private Function InitializePopulation()
        InitializePopulation = Nothing
        Dim i As Long
        'Dim j As Long
        For i = 0 To popSize - 1

            ValidFolding = 0
            RandomOrientation(i)


            While (ValidFolding = 0)
                RandomOrientation(i)
            End While

            'DrawStructurePictureBox1(i, population)

            population(i).Fitness = ComputeFitness(i)

        Next i


    End Function
    ' ================================ End of Initializing population ============================================


    ' ================================ Start of Drawing Structure into Picture Box 2 ============================================
    Public Sub DrawStructurePictureBox2(n As Integer, struct() As genotype)
        Me.PictureBox1.Refresh()
        Dim myGraphics As Graphics
        Dim myPen As New Pen(Color.Black, 2)

        'return the PictureBox1 as a drawing surface
        myGraphics = Graphics.FromHwnd(PictureBox1.Handle)

        Dim myRectangle1 As New Rectangle
        Dim myRectangle2 As New Rectangle
        Dim x As Long = 300         ' Shift the origin by 300 * 150 pixels
        Dim y As Long = 150
        Dim i As Integer
        Dim j As Integer = 0


        For i = 0 To proteinLength - 2
            j = i + 1
            Dim x0 = struct(n).X(i)
            Dim y0 = struct(n).Y(i)
            Dim x1 = struct(n).X(j)
            Dim y1 = struct(n).Y(j)
            x0 *= 15                    ' represent 15 pixel of the picture box as one pixel
            x0 += x                     ' transfer each coordinate to the shifted origin ( which is at 300*200)
            y0 *= 15
            y0 += y

            x1 *= 15
            x1 += x
            y1 *= 15
            y1 += y

            If i = 0 Then
                myRectangle1.X = x0 - 4             ' to draw a circle first consider a rectangle of width 8 and height 8
                myRectangle1.Y = y0 - 4
                myRectangle1.Width = 8
                myRectangle1.Height = 8

                myRectangle2.X = x1 - 4
                myRectangle2.Y = y1 - 4
                myRectangle2.Width = 8
                myRectangle2.Height = 8

                Dim H_index = Array.IndexOf(HPModel, i)
                If H_index >= 0 Then
                    myGraphics.FillEllipse(Brushes.Black, myRectangle1)   ' FillEclipse method draws a circle within the rectange with diameter of 8  
                Else

                    myGraphics.FillEllipse(Brushes.Gray, myRectangle1)    ' H is represented by Black Color ' P is represented by Gray Color

                End If

                Dim H_index_Next = Array.IndexOf(HPModel, j)
                If H_index_Next >= 0 Then
                    myGraphics.FillEllipse(Brushes.Black, myRectangle2)
                Else
                    myGraphics.FillEllipse(Brushes.Gray, myRectangle2)

                End If


            End If

            If i > 0 Then
                myRectangle2.X = x1 - 4
                myRectangle2.Y = y1 - 4
                myRectangle2.Width = 8
                myRectangle2.Height = 8

                Dim H_index_Next = Array.IndexOf(HPModel, j)
                If H_index_Next >= 0 Then
                    myGraphics.FillEllipse(Brushes.Black, myRectangle2)
                Else
                    myGraphics.FillEllipse(Brushes.Gray, myRectangle2)

                End If

            End If

            myGraphics.DrawLine(myPen, x0, y0, x1, y1)      ' Draw the line between two shifted points

        Next

    End Sub
    ' ================================ End of Drawing Strucutre into Picture Box 2 ============================================


    ' ================================ Start of Compute Fitness for each chromosome ============================================
    Private Function ComputeFitness(n As Long)

        Dim F, i, j, TestF, TestSeq As Long

        F = 0
        For i = 0 To HPModel.Length - 2
            For j = i + 1 To HPModel.Length - 1
                TestSeq = (Abs(HPModel(i) - HPModel(j))) '/*Not Sequential */
                If (TestSeq <> 1) Then
                    TestF = Abs(population(n).X(HPModel(i)) - population(n).X(HPModel(j))) + Abs(population(n).Y(HPModel(i)) - population(n).Y(HPModel(j)))
                    If (TestF = 1) Then
                        F = F + 1
                    End If
                End If
            Next j
        Next i

        ComputeFitness = F

    End Function

    ' ================================ End of Compute Fitness for each chromosome ============================================


    ' ================================ Start of Computing random orientation while initialization ============================================
    ' this will generate random orientation for population
    Private Function RandomOrientation(m As Long)
        RandomOrientation = Nothing
        Dim PreviousDirection, PresentDirection As Long
        Dim i, temp1, temp2, temp3, X, Y, j, Flag, Step2 As Long
        Dim rdm As New Random()
        ReDim population(m).X(proteinLength)
        ReDim population(m).Y(proteinLength)
        Dim a(3) As Long
        Dim Ax(3) As Long
        Dim Ay(3) As Long

        '                                        2
        '             Select Direction as:     1 X 0
        '                                        3
        '

        ValidFolding = 1
        population(m).X(0) = 0
        population(m).Y(0) = 0
        population(m).X(1) = 1
        population(m).Y(1) = 0
        PreviousDirection = 0


        For i = 2 To proteinLength - 1

            Select Case PreviousDirection
                Case 0
                    a(0) = 0
                    Ax(0) = 1
                    Ay(0) = 0
                    a(1) = 2
                    Ax(1) = 0
                    Ay(1) = 1
                    a(2) = 3
                    Ax(2) = 0
                    Ay(2) = -1
                Case 1
                    a(0) = 1
                    Ax(0) = -1
                    Ay(0) = 0
                    a(1) = 2
                    Ax(1) = 0
                    Ay(1) = 1
                    a(2) = 3
                    Ax(2) = 0
                    Ay(2) = -1
                Case 2
                    a(0) = 0
                    Ax(0) = 1
                    Ay(0) = 0
                    a(1) = 1
                    Ax(1) = -1
                    Ay(1) = 0
                    a(2) = 2
                    Ax(2) = 0
                    Ay(2) = 1
                Case 3
                    a(0) = 0
                    Ax(0) = 1
                    Ay(0) = 0
                    a(1) = 1
                    Ax(1) = -1
                    Ay(1) = 0
                    a(2) = 3
                    Ax(2) = 0
                    Ay(2) = -1
            End Select

            temp1 = rdm.Next(0, 2)
            'temp1 = Int(2 * Rnd() + 0)
            PresentDirection = temp1
            temp2 = 0
            temp3 = 0
            X = population(m).X(i - 1) + Ax(temp1)
            Y = population(m).Y(i - 1) + Ay(temp1)
            Flag = 0

            For j = 0 To i - 1
                If (X = population(m).X(j) And Y = population(m).Y(j)) Then
                    Flag = 1
                    GoTo MyJump1
                End If
            Next j

MyJump1:
            If (Flag = 1) Then
                Flag = 0
                Step2 = 4 - temp1
                Select Case Step2
                    Case 2
                        If rdm.Next(0, 1) = 0 Then
                            temp2 = 0
                        Else
                            temp2 = 1
                        End If
                    Case 3
                        If rdm.Next(0, 1) = 0 Then
                            temp2 = 0
                        Else
                            temp2 = 2
                        End If
                    Case 4
                        If rdm.Next(0, 1) = 0 Then
                            temp2 = 1
                        Else
                            temp2 = 2
                        End If
                End Select

                PresentDirection = temp2
                temp3 = 4 - (temp1 + temp2)
                X = population(m).X(i - 1) + Ax(temp2)
                Y = population(m).Y(i - 1) + Ay(temp2)

                For j = 0 To i - 1
                    If (X = population(m).X(j) And Y = population(m).Y(j)) Then
                        Flag = 1
                        GoTo MyJump2
                    End If
                Next j
MyJump2:
                If (Flag = 1) Then
                    Flag = 0
                    PresentDirection = temp3
                    X = population(m).X(i - 1) + Ax(temp3)
                    Y = population(m).Y(i - 1) + Ay(temp3)
                    For j = 0 To i - 1
                        If (X = population(m).X(j) And Y = population(m).Y(j)) Then
                            Flag = 1
                            ValidFolding = 0
                            'GoTo MyJump3

                        End If
                    Next j
                End If
            End If
            PreviousDirection = a(PresentDirection)
            population(m).X(i) = population(m).X(i - 1) + Ax(PresentDirection)
            population(m).Y(i) = population(m).Y(i - 1) + Ay(PresentDirection)
        Next i
MyJump3:

    End Function
    ' ================================ End of Computing random orientation while initialization ============================================


    ' ================================ Start of Computing random orientation while for filling in rest ============================================
    ' this will generate random orientation for newpopulation
    Private Function NewRandomOrientation(m As Long)
        NewRandomOrientation = Nothing
        Dim PreviousDirection, PresentDirection As Long
        Dim i, temp1, temp2, temp3, X, Y, j, Flag, Step2 As Long
        Dim rdm As New Random()
        ReDim newpopulation(m).X(proteinLength - 1)
        ReDim newpopulation(m).Y(proteinLength - 1)
        Dim a(3) As Long
        Dim Ax(3) As Long
        Dim Ay(3) As Long

        '                                        2
        '             Select Direction as:     1 X 0
        '                                        3
        '

        ValidFolding = 1
        newpopulation(m).X(0) = 0
        newpopulation(m).Y(0) = 0
        newpopulation(m).X(1) = 1
        newpopulation(m).Y(1) = 0
        PreviousDirection = 0


        For i = 2 To proteinLength - 1

            Select Case PreviousDirection
                Case 0
                    a(0) = 0
                    Ax(0) = 1
                    Ay(0) = 0
                    a(1) = 2
                    Ax(1) = 0
                    Ay(1) = 1
                    a(2) = 3
                    Ax(2) = 0
                    Ay(2) = -1
                Case 1
                    a(0) = 1
                    Ax(0) = -1
                    Ay(0) = 0
                    a(1) = 2
                    Ax(1) = 0
                    Ay(1) = 1
                    a(2) = 3
                    Ax(2) = 0
                    Ay(2) = -1
                Case 2
                    a(0) = 0
                    Ax(0) = 1
                    Ay(0) = 0
                    a(1) = 1
                    Ax(1) = -1
                    Ay(1) = 0
                    a(2) = 2
                    Ax(2) = 0
                    Ay(2) = 1
                Case 3
                    a(0) = 0
                    Ax(0) = 1
                    Ay(0) = 0
                    a(1) = 1
                    Ax(1) = -1
                    Ay(1) = 0
                    a(2) = 3
                    Ax(2) = 0
                    Ay(2) = -1
            End Select

            temp1 = rdm.Next(0, 2)
            'temp1 = Int(2 * Rnd() + 0)
            PresentDirection = temp1
            temp2 = 0
            temp3 = 0
            X = newpopulation(m).X(i - 1) + Ax(temp1)
            Y = newpopulation(m).Y(i - 1) + Ay(temp1)
            Flag = 0

            For j = 0 To i - 1
                If (X = newpopulation(m).X(j) And Y = newpopulation(m).Y(j)) Then
                    Flag = 1
                    GoTo MyJump1
                End If
            Next j

MyJump1:
            If (Flag = 1) Then
                Flag = 0
                Step2 = 4 - temp1
                Select Case Step2
                    Case 2
                        If rdm.Next(0, 1) = 0 Then
                            temp2 = 0
                        Else
                            temp2 = 1
                        End If
                    Case 3
                        If rdm.Next(0, 1) = 0 Then
                            temp2 = 0
                        Else
                            temp2 = 2
                        End If
                    Case 4
                        If rdm.Next(0, 1) = 0 Then
                            temp2 = 1
                        Else
                            temp2 = 2
                        End If
                End Select

                PresentDirection = temp2
                temp3 = 4 - (temp1 + temp2)
                X = newpopulation(m).X(i - 1) + Ax(temp2)
                Y = newpopulation(m).Y(i - 1) + Ay(temp2)

                For j = 0 To i - 1
                    If (X = newpopulation(m).X(j) And Y = newpopulation(m).Y(j)) Then
                        Flag = 1
                        GoTo MyJump2
                    End If
                Next j
MyJump2:
                If (Flag = 1) Then
                    Flag = 0
                    PresentDirection = temp3
                    X = newpopulation(m).X(i - 1) + Ax(temp3)
                    Y = newpopulation(m).Y(i - 1) + Ay(temp3)
                    For j = 0 To i - 1
                        If (X = newpopulation(m).X(j) And Y = newpopulation(m).Y(j)) Then
                            Flag = 1
                            ValidFolding = 0
                            'GoTo MyJump3

                        End If
                    Next j
                End If
            End If
            PreviousDirection = a(PresentDirection)
            newpopulation(m).X(i) = newpopulation(m).X(i - 1) + Ax(PresentDirection)
            newpopulation(m).Y(i) = newpopulation(m).Y(i - 1) + Ay(PresentDirection)
        Next i
MyJump3:

    End Function

    Private Sub Label2_Click(sender As Object, e As EventArgs) Handles Label2.Click

    End Sub

    Private Sub TextBox1_TextChanged(sender As Object, e As EventArgs) Handles TextBox1.TextChanged

    End Sub

    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load

    End Sub

    Private Sub OpenFileDialog1_FileOk(sender As Object, e As System.ComponentModel.CancelEventArgs) Handles OpenFileDialog1.FileOk

    End Sub

    Private Sub Label3_Click(sender As Object, e As EventArgs) Handles Label3.Click

    End Sub

    Private Sub TextBox2_TextChanged(sender As Object, e As EventArgs) Handles TextBox2.TextChanged

    End Sub

    Private Sub Label1_Click(sender As Object, e As EventArgs) Handles Label1.Click

    End Sub
    ' ================================ End of Computing random orientation while for filling in rest ============================================


End Class

