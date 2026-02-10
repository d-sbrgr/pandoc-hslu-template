-- Lua filter to split document into main content, appendix, and bibliography
-- Main content goes in $body$
-- Appendix content goes to $appendix$ variable in metadata
-- Bibliography content goes to $bibliography$ variable in metadata
-- This allows us to place glossary and other back matter between them in the template

function Pandoc(doc)
  local main = {}
  local abstract = {}
  local appendix = {}
  local bibliography = {}
  local in_appendix = false
  local in_bibliography = false
  local in_abstract = false

  io.stderr:write("DEBUG: Starting split-appendix filter\n")

  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then
      local text = pandoc.utils.stringify(block.content)
      io.stderr:write("DEBUG: Found header level 1: '" .. text .. "'\n")
      -- Check for Appendix heading (unnumbered with {-})
      if text:match("^Appendix%s*$") or text == "Appendix" then
        io.stderr:write("DEBUG: Matched Appendix heading!\n")
        in_appendix = true
        in_bibliography = false
        in_abstract = false
        -- Start appendix section with the header
        table.insert(appendix, pandoc.RawBlock("latex", "\\appendix"))
        -- Add chapter heading for appendix
        table.insert(appendix, block)
      -- Check for Bibliography/References heading (unnumbered with {-})
      elseif text:match("^Bibliography%s*$") or text == "Bibliography" or text:match("^References%s*$") or text == "References" then
        io.stderr:write("DEBUG: Matched Bibliography heading!\n")
        in_bibliography = true
        in_appendix = false
        in_abstract = false
        -- Don't add raw latex command, just add the header
        table.insert(bibliography, block)
      elseif text:match("^Abstract%s*$") or text == "Abstract" then
        io.stderr:write("DEBUG: Matched Abstract heading!\n")
        in_abstract = true
        in_appendix = false
        in_bibliography = false
        table.insert(abstract, block)
      else
        in_bibliography = false
        in_appendix = false
        in_abstract = false
        table.insert(main, block)
      end
    elseif in_appendix then
      table.insert(appendix, block)
    elseif in_bibliography then
      table.insert(bibliography, block)
    elseif in_abstract then
      table.insert(abstract, block)
    else
      table.insert(main, block)
    end
  end

  io.stderr:write("DEBUG: Main blocks: " .. #main .. ", Appendix blocks: " .. #appendix .. ", Bibliography blocks: " .. #bibliography ..", Abstract blocks: " .. #abstract .."\n")

  -- Store appendix content in metadata if it exists
  if #appendix > 0 then
    doc.meta.appendix = pandoc.MetaBlocks(appendix)
    io.stderr:write("DEBUG: Appendix content stored in metadata\n")
  end

  -- Store bibliography content in metadata if it exists
  if #bibliography > 0 then
    doc.meta.bibliography = pandoc.MetaBlocks(bibliography)
    io.stderr:write("DEBUG: Bibliography content stored in metadata\n")
  end

  -- Store abstract content in metadata if it exists
  if #abstract > 0 then
      doc.meta.abstract = pandoc.MetaBlocks(abstract)
      io.stderr:write("DEBUG: Abstract content stored in metadata\n")
  end

  -- Return document with only main content
  doc.blocks = main
  return doc
end

