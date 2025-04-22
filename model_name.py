import matplotlib.pyplot as plt
import math

# Full model list
model_names = [
    "AlbertForMaskedLM", "AlbertForQuestionAnswering", "AllenaiLongformerBase",
    "BartForCausalLM", "BartForConditionalGeneration", "BertForMaskedLM",
    "BertForQuestionAnswering", "BigBird", "BlenderbotForCausalLM",
    "BlenderbotSmallForCausalLM", "BlenderbotSmallForConditionalGeneration", "CamemBert",
    "DebertaForMaskedLM", "DebertaForQuestionAnswering", "DebertaV2ForMaskedLM",
    "DebertaV2ForQuestionAnswering", "DistilBertForMaskedLM", "DistilBertForQuestionAnswering",
    "DistilGPT2", "ElectraForCausalLM", "ElectraForQuestionAnswering",
    "GoogleFnet", "GPT2ForSequenceClassification", "LayoutLMForMaskedLM",
    "LayoutLMForSequenceClassification", "M2M100ForConditionalGeneration", "MBartForConditionalGeneration",
    "MegatronBertForCausalLM", "MegatronBertForQuestionAnswering", "MobileBertForMaskedLM",
    "MobileBertForQuestionAnswering", "MT5ForConditionalGeneration", "OPTForCausalLM",
    "PegasusForCausalLM", "PegasusForConditionalGeneration", "PLBartForConditionalGeneration",
    "RobertaForCausalLM", "RobertaForQuestionAnswering", "Speech2Text2ForCausalLM",
    "T5ForConditionalGeneration", "T5Small", "TrOCRForCausalLM",
    "XGLMForCausalLM", "XLNetLMHeadModel", "YituTechConvBert"
]

# Grid dimensions
n_cols = 3
n_rows = math.ceil(len(model_names) / n_cols)

# Pad to fill last row
while len(model_names) < n_cols * n_rows:
    model_names.append("")

# Organize into 2D table
table_data = [model_names[i:i + n_cols] for i in range(0, len(model_names), n_cols)]

# Estimate column widths based on max string length in each column
max_lengths = [max(len(row[i]) for row in table_data) for i in range(n_cols)]
total_length = sum(max_lengths)
col_widths = [l / total_length for l in max_lengths]  # normalize

# Plotting
fig, ax = plt.subplots(figsize=(18, n_rows * 0.9))  # wider for fewer columns
ax.axis("off")

# Create table
table = ax.table(cellText=table_data,
                 cellLoc='center',
                 loc='center',
                 colWidths=col_widths)

table.auto_set_font_size(False)
table.set_fontsize(15)
table.scale(1, 1.5)

# Style
for cell in table.get_celld().values():
    cell.set_linewidth(0.5)

plt.tight_layout()
plt.savefig("model_table_3col.png")
plt.show()
