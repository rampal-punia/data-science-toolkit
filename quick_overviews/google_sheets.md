# Comprehensive Google Sheets Tutorial

# Basic (Day-to-Day Tasks)

## 1. Comments

### 1.1 Adding Comments
- Select a cell or range
- Right-click > Insert comment
- Or use keyboard shortcut: Ctrl+Alt+M
- Type your comment and click 'Comment'

### 1.2 Viewing Comments
- Cells with comments have a small orange triangle in the top-right corner
- Hover over the cell to view the comment
- Click the comment to open it fully

### 1.3 Replying to Comments
- Open a comment
- Click 'Reply' at the bottom
- Type your reply and click 'Reply'

### 1.4 Resolving Comments
- Open a comment
- Click the checkbox icon to mark it as resolved
- Resolved comments can be viewed by clicking 'Comments' in the top-right corner

## 2. Notes

### 2.1 Adding Notes
- Select a cell
- Insert > Note
- Or use keyboard shortcut: Shift+F2
- Type your note and click outside the cell

### 2.2 Viewing Notes
- Cells with notes have a small black triangle in the top-right corner
- Hover over the cell to view the note

### 2.3 Editing Notes
- Right-click a cell with a note > Edit note
- Make changes and click outside the cell to save

## 3. Named Ranges

### 3.1 Creating Named Ranges
- Select a range of cells
- Data > Named ranges
- Enter a name for the range and click 'Done'

### 3.2 Using Named Ranges
- In formulas, use the range name instead of cell references
- Example: `=SUM(MonthlyExpenses)` instead of `=SUM(A1:A12)`

### 3.3 Managing Named Ranges
- Data > Named ranges
- View, edit, or delete existing named ranges

## 4. Data Validation

### 4.1 Setting Up Data Validation
- Select a cell or range
- Data > Data validation
- Choose criteria (e.g., list of items, number range)
- Add a custom error message if needed

### 4.2 Using Data Validation for Documentation
- Add dropdown lists for consistent data entry
- Use custom error messages to provide guidance

## 5. Conditional Formatting

### 5.1 Creating Conditional Formatting Rules
- Select a range
- Format > Conditional formatting
- Set up rules to highlight cells based on their content

### 5.2 Using Conditional Formatting for Documentation
- Highlight important data or outliers
- Use color scales to visualize trends

## 6. Sheet and Range Protection

### 6.1 Protecting a Sheet
- Right-click sheet tab > Protect sheet
- Set permissions for other users

### 6.2 Protecting a Range
- Select a range
- Data > Protect sheets and ranges
- Set permissions for the selected range

## 7. Version History

### 7.1 Viewing Version History
- File > Version history > See version history
- Browse through past versions of your sheet

### 7.2 Naming Versions
- In version history, click the three dots next to a version
- Select 'Name this version'
- Enter a descriptive name

### 7.3 Restoring Previous Versions
- In version history, select a version
- Click 'Restore this version'

## 8. Using Sheets for Documentation

### 8.1 Creating a Change Log
- Dedicate a sheet to tracking changes
- Columns: Date, User, Description of Change, Affected Ranges

### 8.2 Data Dictionary
- Create a sheet explaining column meanings, data types, and sources
- Include any relevant business rules or calculations

### 8.3 README Sheet
- Add a sheet named 'README' at the beginning of your workbook
- Include overview, purpose, and instructions for using the spreadsheet

## 9. Linking and References

### 9.1 Cell References Across Sheets
- Use `SheetName!CellReference`
- Example: `=SUM(Expenses!A1:A10)`

### 9.2 Hyperlinks Within the Spreadsheet
- `=HYPERLINK("#gid=sheet_id&range=A1", "Link Text")`
- Replace sheet_id with the actual sheet ID (found in the sheet's URL)

### 9.3 External Links
- `=HYPERLINK("https://example.com", "Link Text")`

## 10. Add-ons for Documentation

### 10.1 Finding Documentation Add-ons
- Add-ons > Get add-ons
- Search for 'documentation' or 'data dictionary'

### 10.2 Popular Documentation Add-ons
- 'Table of Contents': Automatically generate a clickable table of contents
- 'Doc Variables': Create and manage variables across your spreadsheet


## 1. Usefule Functions

### 1.1 VLOOKUP and HLOOKUP
- Syntax: `VLOOKUP(search_key, range, index, [is_sorted])`
- Example: `=VLOOKUP(A2, B2:D10, 2, FALSE)`
- HLOOKUP for horizontal data

### 1.2 INDEX and MATCH
- More flexible than VLOOKUP
- Syntax: `INDEX(range, MATCH(lookup_value, lookup_range, 0))`
- Example: `=INDEX(B2:D10, MATCH(A2, B2:B10, 0), 2)`

### 1.3 QUERY
- Uses SQL-like syntax to manipulate data
- Syntax: `=QUERY(data, query, [headers])`
- Example: `=QUERY(A1:D100, "SELECT A, SUM(B) WHERE C = 'Complete' GROUP BY A", 1)`

## 2. Array Formulas

### 2.1 Basic Array Formulas
- Use Ctrl+Shift+Enter to create array formulas
- Example: `=ArrayFormula(A1:A10 * B1:B10)`

### 2.2 SUMPRODUCT
- Multiplies ranges or arrays and sums the products
- Example: `=SUMPRODUCT(A1:A10, B1:B10)`

### 2.3 Array Constants
- Use curly braces to define array constants
- Example: `={1,2,3; 4,5,6; 7,8,9}`

## 3. Advanced Conditional Formatting

### 3.1 Custom Formulas
- Use formulas to create complex conditional formatting rules
- Example: `=MOD(ROW(),2)=0` to highlight every other row

### 3.2 Conditional Formatting with ARRAYFORMULA
- Apply conditional formatting to entire columns
- Example: `=ARRAYFORMULA($A:$A<TODAY())` to highlight past dates

## 4. Pivot Tables and Charts

### 4.1 Creating Advanced Pivot Tables
- Use calculated fields and custom formulas
- Apply filtering and slicing for data analysis

### 4.2 Pivot Charts
- Create dynamic charts based on pivot table data
- Use slicers for interactive data visualization

## 5. Custom Functions with Apps Script

### 5.1 Writing Custom Functions
- Use Tools > Script editor to open the Apps Script editor
- Example:
  ```javascript
  function DOUBLE(input) {
    return input * 2;
  }
  ```

### 5.2 Using Custom Functions
- Use your function in sheets: `=DOUBLE(A1)`

## 6. Data Cleaning and Transformation

### 6.1 Text Functions
- SPLIT, CONCATENATE, TRIM, CLEAN
- Example: `=SPLIT(A1, " ")` to separate words in a cell

### 6.2 Regular Expressions
- REGEXEXTRACT, REGEXREPLACE, REGEXMATCH
- Example: `=REGEXEXTRACT(A1, "(\d{3})-(\d{3})-(\d{4})")` to extract a phone number

## 7. Data Validation and Dropdown Lists

### 7.1 Dependent Dropdowns
- Create cascading dropdowns using data validation and INDIRECT function
- Example: `=INDIRECT($A$1)` where A1 contains the range name for the dropdown

### 7.2 Custom Data Validation
- Use custom formulas for complex validation rules
- Example: `=COUNTIF($A$1:$A$10, A1)<=1` to ensure unique entries

## 8. Advanced Charts and Graphs

### 8.1 Combo Charts
- Combine different chart types (e.g., column and line)
- Use secondary axis for different scales

### 8.2 Sparklines
- Create mini-charts in a single cell
- Syntax: `=SPARKLINE(data, [options])`
- Example: `=SPARKLINE(A1:A12, {"charttype","line"})`

## 9. Automation with Macros

### 9.1 Recording Macros
- Tools > Macros > Record macro
- Perform actions to be recorded
- Save and name your macro

### 9.2 Running Macros
- Tools > Macros > [Your Macro Name]
- Or assign a custom shortcut key

## 10. Advanced Filters and Sorting

### 10.1 Filter Views
- Create and save multiple filter configurations
- Data > Filter views > Create new filter view

### 10.2 Complex Sorting
- Sort by multiple columns
- Use custom formulas for sorting

## 1. More Functions and Formulas

### 1.1 Sum and Average
- `=SUM(A1:A10)`: Adds up values in cells A1 through A10
- `=AVERAGE(A1:A10)`: Calculates the average of values in cells A1 through A10

### 1.2 Count Functions
- `=COUNT(A1:A10)`: Counts the number of cells with numerical values
- `=COUNTA(A1:A10)`: Counts the number of non-empty cells
- `=COUNTIF(A1:A10, ">10")`: Counts cells meeting a specific condition

### 1.3 IF Statements
- `=IF(A1>10, "High", "Low")`: If A1 is greater than 10, return "High", otherwise "Low"

## 2. Data Manipulation

### 2.1 Text Functions
- `=CONCATENATE(A1, " ", B1)`: Combines text from multiple cells
- `=LEFT(A1, 5)`: Returns the leftmost 5 characters from A1
- `=RIGHT(A1, 5)`: Returns the rightmost 5 characters from A1
- `=MID(A1, 3, 5)`: Returns 5 characters starting from the 3rd position in A1

### 2.2 Date Functions
- `=TODAY()`: Returns today's date
- `=DATEVALUE("1/1/2023")`: Converts a date string to a date value
- `=DATEDIF(A1, B1, "Y")`: Calculates the difference between two dates in years

### 2.3 Lookup Functions
- `=VLOOKUP(A1, B1:C10, 2, FALSE)`: Searches for A1 in the first column of B1:C10 and returns the corresponding value from the second column
- `=INDEX(B1:C10, MATCH(A1, B1:B10, 0), 2)`: A more flexible alternative to VLOOKUP

## 3. Data Analysis

### 3.1 Pivot Tables
1. Select your data range
2. Go to "Insert" > "Pivot table"
3. Drag and drop fields into Rows, Columns, Values, and Filters
4. Use the "Summarize by" option to choose how to aggregate your data (sum, average, count, etc.)

### 3.2 Conditional Formatting
1. Select your data range
2. Go to "Format" > "Conditional formatting"
3. Choose a rule type (e.g., Color scale, Data bar, Icon set)
4. Customize the rule to highlight cells based on their values

### 3.3 Charts
1. Select your data range
2. Go to "Insert" > "Chart"
3. Choose the appropriate chart type
4. Customize using the Chart editor sidebar

## 4. Automation and Productivity

### 4.1 Custom Functions
1. Go to "Tools" > "Script editor"
2. Write a custom function in JavaScript, for example:
   ```javascript
   function DOUBLE(input) {
     return input * 2;
   }
   ```
3. Use in your sheet: `=DOUBLE(A1)`

### 4.2 Data Validation
1. Select a cell or range
2. Go to "Data" > "Data validation"
3. Set up rules to restrict input (e.g., list of items, number range)

### 4.3 Named Ranges
1. Select a range of cells
2. Go to "Data" > "Named ranges"
3. Give your range a name
4. Use the name in formulas instead of cell references

## 5. Collaboration and Sharing

### 5.1 Sharing Settings
1. Click the "Share" button in the top right
2. Add people by email address
3. Choose permission levels (Editor, Commenter, Viewer)

### 5.2 Comments and Suggestions
- Add comments: Select a cell, right-click, and choose "Comment"
- Suggest edits: Turn on "Suggesting" mode in the top right corner

### 5.3 Version History
1. Go to "File" > "Version history" > "See version history"
2. Review past versions and restore if needed

## 6. Advanced Features

### 6.1 Query Function
- Use SQL-like syntax to manipulate data:
  `=QUERY(A1:D100, "SELECT A, B, SUM(C) WHERE D = 'Complete' GROUP BY A, B")`

### 6.2 Array Formulas
- Enter formulas that work on multiple rows/columns at once:
  `=ARRAYFORMULA(A1:A10 * B1:B10)`

### 6.3 Regular Expressions
- Use REGEXEXTRACT for complex text parsing:
  `=REGEXEXTRACT(A1, "(\d{3})-(\d{3})-(\d{4})")`

### 6.4 Google Finance Function
- Pull real-time stock data:
  `=GOOGLEFINANCE("GOOGL", "price")`

## 7. Add-ons and Integrations

### 7.1 Google Sheets Add-ons
1. Go to "Add-ons" > "Get add-ons"
2. Browse and install add-ons for extended functionality

### 7.2 Connecting to Other Google Services
- Use `IMPORTRANGE` to pull data from other Google Sheets
- Connect to Google Forms for automatic data collection
- Use Apps Script to integrate with other Google services like Calendar or Gmail
