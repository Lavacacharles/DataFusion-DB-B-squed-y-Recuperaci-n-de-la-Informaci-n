import math
import pickle
'''
Pasa los diccionarios de términos a listas,
los ordena por término y luego cada posting 
list lo convierte de diccionarios a listas
'''
def OrdenarPorBloques(dir_blocks, n_blocks):
    for i in range(n_blocks):
        file_path = os.path.join(dir_blocks, 'block_{}.pkl'.format(i))
        with open(file_path, 'rb') as f:
            tuplas_ordenadas = pickle.load(f)
        tuplas_ordenadas = sorted(list(tuplas_ordenadas.items()), key=lambda x: x[0])
        tuplas_ordenadas = [par for par in tuplas_ordenadas if not par[0].isdigit()]
        tuplas_ordenadas = [(term[0], list(term[1].items())) for term in file]
        with open(file_path, 'wb') as f:
            pickle.dump(tuplas_ordenadas, f)

def mergeSortAux(dir_bloques, l, r):
    if l == r:
        bloque = leer_bloque(dir_bloques, l)
        unicos = set()    
        for par in bloque:
            unicos.add(par[0])
        return list(unicos)
    
    if (l < r):
        mid = int(math.ceil((r + l)/2.0))
        unique_l = mergeSortAux(dir_bloques, l, mid - 1)
        unique_r = mergeSortAux(dir_bloques, mid, r)
        unicos = set()    
        for term in unique_l:
            unicos.add(term)
        for term in unique_r:
            unicos.add(term)
        unicos = list(unicos)
        merge_v2(dir_bloques, l, r, mid, len(unicos))
        return list(unicos)
        
    return []

def escribir_bloque(dir_bloques, block, idx_insert_block, buffer_limit = 2000):
    with open(os.path.join(dir_bloques, "block_{}_v2.pkl".format(idx_insert_block)), 'wb') as f:
        pickle.dump(block, f)    
def leer_bloque(dir_bloques, it):
    file_path = os.path.join(dir_bloques, f"block_{it}.pkl")
    with open(file_path, "rb") as f:
        buffer = pickle.load(f)
    return buffer
def merge_v2(dir_bloques, l, r, mid, num_terms):
    idx_insert_block = l
    new_block = []
    mezclar_n_bloques = r - l + 1
    unique_terms_per_block = int(math.ceil(num_terms/mezclar_n_bloques))
    unique_terms_current_block = 0

    it_l = l
    it_r = mid
    term_dic_l = leer_bloque(dir_bloques, it_l)
    term_dic_r = leer_bloque(dir_bloques, it_r)
    
    idx_term_l = 0
    idx_term_r = 0

    idx_doc_l = 0
    idx_doc_r = 0
    new_block = []
    while(it_l < mid and it_r < r + 1):
        print(f"Toma 2 bloques {it_l} y {it_r} | idx_term_l: ", idx_term_l, "| len(term_dic_l)", len(term_dic_l), "| idx_term_r: ", idx_term_r, "| len(term_dic_r)", len(term_dic_r))
        while(idx_term_l < len(term_dic_l) and idx_term_r < len(term_dic_r)): # moverme entre palabras de dos bloques
            print(f"Toma 2 terminos {term_dic_l[idx_term_l][0]} y {term_dic_r[idx_term_r][0]}")
            new_term = []
            if(term_dic_l[idx_term_l][0] < term_dic_r[idx_term_r][0]):
                new_term = term_dic_l[idx_term_l]
                idx_term_l += 1
            elif(term_dic_l[idx_term_l][0] > term_dic_r[idx_term_r][0]):
                new_term = term_dic_r[idx_term_r]
                idx_term_r += 1
            else:
                idx_doc_l = 0
                idx_doc_r = 0
                while(idx_doc_l < len(term_dic_l[idx_term_l][1]) and idx_doc_r < len(term_dic_r[idx_term_r][1])):
                    print(f"Toma 2 terminos iguales con tf = {term_dic_l[idx_term_l][1]} y {term_dic_r[idx_term_r][1]}")
                    if term_dic_l[idx_term_l][1][idx_doc_l][0] > term_dic_r[idx_term_r][1][idx_doc_r][0]:
                        pushear_doc = term_dic_r[idx_term_r][1][idx_doc_r]
                        idx_doc_r += 1
                    elif term_dic_l[idx_term_l][1][idx_doc_l][0] < term_dic_r[idx_term_r][1][idx_doc_r][0]:
                        pushear_doc = term_dic_l[idx_term_l][1][idx_doc_l]
                        idx_doc_l += 1
                    else:
                        pushear_doc = (term_dic_l[idx_term_l][1][idx_doc_l][0], term_dic_l[idx_term_l][1][idx_doc_l][1] + term_dic_r[idx_term_r][1][idx_doc_r][1])
                        idx_doc_l += 1
                        idx_doc_r += 1
                    print("pushear_doc: ", pushear_doc)
                    new_term.append(pushear_doc)
                while(idx_doc_l < len(term_dic_l[idx_term_l][1])):
                    print(f"ya no hay documentos de derecha, rellena con izquierda {idx_doc_l}")
                    pushear_doc = term_dic_l[idx_term_l][1][idx_doc_l]
                    idx_doc_l += 1
                    new_term.append(pushear_doc)
                while(idx_doc_r < len(term_dic_r[idx_term_r][1])):
                    print(f"ya no hay documentos de izquierda, rellena con derecha {idx_doc_l}")
                    pushear_doc = term_dic_r[idx_term_r][1][idx_doc_r]
                    idx_doc_r += 1
                    new_term.append(pushear_doc)
                new_term = (term_dic_l[idx_term_l][0], new_term)
                idx_term_r += 1
                idx_term_l += 1
            new_block.append(new_term)
            
            unique_terms_current_block += 1
            if (unique_terms_current_block == unique_terms_per_block):
                escribir_bloque(dir_bloques, new_block, idx_insert_block)
                unique_terms_current_block = 0
                idx_insert_block += 1
                new_block = []
        if(len(term_dic_l) == idx_term_l):
            if (it_l < mid - 1):
                it_l += 1
                term_dic_l = leer_bloque(dir_bloques, it_l)
                idx_term_l = 0
                idx_doc_l = 0
                continue
            else:
                break
        if(len(term_dic_r) == idx_term_r):
            if (it_r < r):
                it_r += 1
                term_dic_r = leer_bloque(dir_bloques, it_r)
                idx_term_r = 0
                idx_doc_r = 0
                continue
            else:
                break
        if(it_l == mid | it_r == r + 1):
            break
    while(it_l < mid):
        print(f"Se acabaron los bloques de derecha, llena solo izquierda {it_l}")
        term_dic_l = leer_bloque(dir_bloques, it_l)
        while(idx_term_l < len(term_dic_l)):
            new_block.append(term_dic_l[idx_term_l])
            unique_terms_current_block += 1
            if (unique_terms_current_block == unique_terms_per_block):
                escribir_bloque(dir_bloques, new_block, idx_insert_block)
                unique_terms_current_block = 0
                idx_insert_block += 1
                new_block = []
            idx_term_l += 1
        idx_term_l = 0
        it_l += 1
    while(it_r < r + 1):
        print(f"Se acabaron los bloques de izquierda, llena solo derecha {it_r}")
        term_dic_r = leer_bloque(dir_bloques, it_r)
        while(idx_term_r < len(term_dic_r)):
            new_block.append(term_dic_r[idx_term_r])
            unique_terms_current_block += 1
            if (unique_terms_current_block == unique_terms_per_block):
                escribir_bloque(dir_bloques, new_block, idx_insert_block)
                unique_terms_current_block = 0
                idx_insert_block += 1
                new_block = []
            idx_term_r += 1
        idx_term_r = 0
        it_r += 1
        


    while(idx_insert_block < r + 1):
        if len(new_block) > 0:
            escribir_bloque(dir_bloques, new_block, idx_insert_block)
        else:
            escribir_bloque(dir_bloques, [], idx_insert_block)
        new_block = []
        idx_insert_block += 1
    idx_insert_block = l
    for idx_archivo in range(l, r + 1):
        nuevo_nombre = os.path.join(dir_bloques, "block_{}.pkl".format(idx_archivo))
        if os.path.exists(nuevo_nombre):
            os.remove(nuevo_nombre)
        os.rename(os.path.join(dir_bloques, "block_{}_v2.pkl".format(idx_archivo)), nuevo_nombre)


def mergeSort(dir_bloques):
    bloques_files_dir = os.listdir(os.path.join('./',dir_bloques))
    n = len(bloques_files_dir)
    # n = int(math.exp2(math.floor(math.log2(n)) + 1))
    mergeSortAux(dir_bloques, 0, n - 1)
'''
Recibe la direccion del folder con los diccionarios del spimi,
modifica los archivos y los sobreescribe para crear el índice
invertido con un índice global
'''
dir_blocks = "index_blocks_v2"
n_blocks = s.block_counter
dir_blocks = 'index_blocks_v2'
mergeSort(dir_blocks)